# 多功能
from pathlib import Path
from typing import NotRequired, List, TypedDict, Callable, Optional, Dict
import dask.dataframe as dd
from dask.distributed import Client, wait
import pandas as pd
import numpy as np
import gc
import psutil
import re


class StandardSetting(TypedDict):
    column: List[str]
    standard_function: NotRequired[Callable[[pd.Series], pd.Series]]
    multiple_judgement_function: NotRequired[Callable[[pd.DataFrame], pd.Series]]
    feature_engineering_function: NotRequired[Callable[[pd.DataFrame], pd.DataFrame]]


class ReconstructionDataSet(TypedDict):
    standard: Optional[List[StandardSetting]]
    multiple_judgement: Optional[List[StandardSetting]]
    feature_engineering: Optional[List[StandardSetting]]
    reserve_column: bool


class ManagerSetting(TypedDict):
    path: Path
    settings: List[ReconstructionDataSet]
    blocksize: str
    compression: NotRequired[str]
    output_format: NotRequired[str]


class DataManager:
    """支持TB级数据处理和智能内存管理的增强版数据管理器"""

    def __init__(self, setting: ManagerSetting, n_workers: int = 4, memory_limit: str = '1GB') -> None:
        self.setting = setting
        self.client = Client(n_workers=n_workers, memory_limit=memory_limit)

        self.ddf = None
        self._memory_limit_gb = self._parse_memory_limit(memory_limit)
        self._original_columns = []

        # 初始化时加载数据
        self._load_data()

    @staticmethod
    def _parse_memory_limit(limit_str: str) -> float:
        """解析内存限制字符串为GB单位"""
        units = {'B': 1e-9, 'KB': 1e-6, 'MB': 1e-3, 'GB': 1, 'TB': 1e3}
        match = re.match(r'^(\d+)([A-Za-z]+)$', limit_str.strip())
        if not match:
            raise ValueError(f"Invalid memory limit format: {limit_str}")
        value, unit = match.groups()
        return float(value) * units[unit.upper()]

    def _load_data(self):
        """加载数据并自动推断文件格式"""
        path = str(self.setting['path'])
        blocksize = self.setting.get('blocksize', '256MB')

        if path.endswith('.parquet'):
            self.ddf = dd.read_parquet(path)
        elif path.endswith('.csv'):
            self.ddf = dd.read_csv(path, blocksize=blocksize)
        elif path.endswith('.json'):
            self.ddf = dd.read_json(path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {path.split('.')[-1]}")

        self._original_columns = list(self.ddf.columns)
        self._persist_data()

    def _persist_data(self):
        """持久化数据到集群内存并等待完成"""
        if self.ddf is not None:
            self.ddf = self.ddf.persist()
            wait(self.ddf)

    def smart_gc(self, force: bool = False):
        """智能内存回收机制"""
        workers_mem = self.client.run(lambda: psutil.Process().memory_info().rss / (1024 ** 3))
        threshold = 0.75 * self._memory_limit_gb

        if force or any(p > threshold for p in workers_mem.values()):
            self.client.cancel(self.client.futures)
            self.client.run(lambda: gc.collect(generation=2))

            if any(p > 0.9 * self._memory_limit_gb for p in workers_mem.values()):
                self.client.restart()
                self.ddf = None  # 清除旧引用
                self._load_data()  # 重启后重新加载数据
            return True
        return False

    def get_processed_columns(self) -> List[str]:
        """获取当前处理后的列列表"""
        return list(self.ddf.columns) if self.ddf is not None else []

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, 'client'):
            self.client.close()


class DataReconstruction:
    """支持分布式处理的大数据重构引擎"""

    def __init__(self, manager: DataManager):
        self.manager = manager
        self._current_processed = set()

    def process_all(self):
        """执行所有重构设置"""
        for recon_set in self.manager.setting['settings']:
            self.process_dataset(recon_set)

    def process_dataset(self, recon_set: ReconstructionDataSet):
        """处理单个重构数据集"""
        # 执行标准化处理
        if recon_set['standard']:
            for setting in recon_set['standard']:
                self._apply_standard(setting)

        # 执行多重判断过滤
        if recon_set['multiple_judgement']:
            for setting in recon_set['multiple_judgement']:
                self._apply_judgment(setting)

        # 执行特征工程
        if recon_set['feature_engineering']:
            for setting in recon_set['feature_engineering']:
                self._apply_feature_engineering(setting)

        # 处理列保留逻辑
        if not recon_set['reserve_column']:
            self._cleanup_columns()

    def _apply_standard(self, setting: StandardSetting):
        """应用列标准化函数"""
        cols = setting['column']
        func = setting.get('standard_function')

        if func:
            for col in cols:
                new_col = f"{col}_standardized"  # 生成新列名
                # 使用map_partitions进行分布式处理
                self.manager.ddf[new_col] = self.manager.ddf[col].map_partitions(
                    func, meta=(new_col, self.manager.ddf[col].dtype))
                self._current_processed.add(new_col)

                self.manager._persist_data()
                self.manager.smart_gc()

    def _apply_judgment(self, setting: StandardSetting):
        """应用多重判断过滤"""
        cols = setting['column']
        func = setting.get('multiple_judgement_function')

        if func:
            # 生成分布式布尔掩码
            mask = self.manager.ddf[cols].map_partitions(
                lambda df: func(df), meta=('mask', bool)
            )

            # 过滤数据
            self.manager.ddf = self.manager.ddf[mask]

            # 新增重分区逻辑（参考网页1[1](@ref)）
            self.manager.ddf = self.manager.ddf.repartition(partition_size="100MB").persist()

            # 关键优化2：索引重置（网页9[9](@ref)）
            self.manager.ddf = self.manager.ddf.reset_index(drop=True).persist()

            # 内存检查点（网页4[4](@ref)）
            if self.manager.smart_gc():
                self.manager.ddf = self.manager.ddf.persist()

            wait(self.manager.ddf)

    def _apply_feature_engineering(self, setting: StandardSetting):
        """应用特征工程函数"""
        cols = setting['column']
        func = setting.get('feature_engineering_function')

        if func:
            # 生成新特征
            new_features = self.manager.ddf[cols].map_partitions(
                func, meta=self._infer_feature_meta(func)
            ).reset_index(drop=True)  # 关键修改：重置索引

            # 合并新特征到主数据集
            self.manager.ddf = dd.concat(
                [self.manager.ddf.reset_index(drop=True), new_features],
                axis=1
            )  # 主数据集也重置索引
            self._current_processed.update(new_features.columns)

            self.manager._persist_data()
            self.manager.smart_gc()

    def _cleanup_columns(self):
        """清理原始列（当reserve_column=False时调用）"""
        current_columns = set(self.manager.ddf.columns) if self.manager.ddf is not None else set()
        cols_to_drop = [col for col in self.manager._original_columns if col in current_columns]
        if cols_to_drop:
            self.manager.ddf = self.manager.ddf.drop(columns=cols_to_drop)
            self.manager._persist_data()

    @staticmethod
    def _infer_feature_meta(func: Callable) -> Dict[str, np.dtype]:
        """通过示例数据推断新特征的元数据"""
        sample_df = pd.DataFrame({f'col{i}': [0] for i in range(5)})
        result = func(sample_df)
        return {col: result[col].dtype for col in result.columns}


# 示例用法
"""
if __name__ == "__main__":
    # 定义标准化函数
    def standardize(s: pd.Series) -> pd.Series:
        return (s - s.mean()) / s.std()


    # 定义多重判断函数
    def quality_check(df: pd.DataFrame) -> pd.Series:
        return (df['value'] > 0) & (df['temperature'] < 100)


    # 定义特征工程函数
    def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            'value_log': np.log1p(df['value']),
            'temp_squared': df['temperature'] ** 2
        })


    # 配置处理参数
    settings = ManagerSetting(
        path=Path('bigdata.csv'),
        blocksize='256MB',
        settings=[
            ReconstructionDataSet(
                standard=[
                    StandardSetting(
                        column=['value', 'temperature'],
                        standard_function=standardize
                    )
                ],
                multiple_judgement=[
                    StandardSetting(
                        column=['value', 'temperature'],
                        multiple_judgement_function=quality_check
                    )
                ],
                feature_engineering=[
                    StandardSetting(
                        column=['value', 'temperature'],
                        feature_engineering_function=feature_engineer
                    )
                ],
                reserve_column=False
            )
        ]
    )

    # 初始化数据管理器
    manager = DataManager(settings)

    try:
        # 执行数据重构
        processor = DataReconstruction(manager)
        processor.process_all()
    finally:
        manager.close()  # 确保资源释放
"""