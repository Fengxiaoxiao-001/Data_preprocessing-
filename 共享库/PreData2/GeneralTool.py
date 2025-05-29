# 工具
import dask.dataframe as dd
from dask.distributed import Client, wait
from pathlib import Path
from typing import TypedDict, List, Union, Optional, Literal
import numpy as np
import pandas as pd
import gc
import re



class EnhancedSettings(TypedDict):
    path: Path
    blocksize: str
    na_value: Union[str, List[str]]
    dtype: dict
    compression: str
    output_format: Literal['csv', 'parquet']
    save_path: Optional[str]
    time_column: Optional[str]
    time_interval: Optional[str]


class EnhancedDataManager:
    """融合版数据管理器（支持TB级数据处理和多任务协同）"""

    def __init__(self, settings: EnhancedSettings,
                 n_workers: int = 4,
                 memory_limit: str = '1GB',
                 radical_optimization: bool = False):

        self.settings = settings
        self.client = Client(n_workers=n_workers, memory_limit=memory_limit)
        self.ddf = None
        self.gc_counter = 0
        self.max_gc_counter = 3
        self.memory_limit_gb = self._parse_memory_limit(memory_limit)

        if radical_optimization:
            self._apply_aggressive_config()

    def _parse_memory_limit(self, limit_str: str) -> float:
        """智能内存限制解析"""
        units = {'B': 1e-9, 'KB': 1e-6, 'MB': 1e-3, 'GB': 1, 'TB': 1e3}
        match = re.match(r'^(\d+)([A-Za-z]+)$', limit_str.strip())
        if not match:
            raise ValueError(f"Invalid memory limit format: {limit_str}")
        value, unit = match.groups()
        return float(value) * units[unit.upper()]

    def _apply_aggressive_config(self):
        """性能激进模式配置"""
        from dask.config import set
        set({
            "optimization.fuse.active": True,
            "distributed.worker.memory.target": 0.6,
            "distributed.worker.memory.spill": 0.75,
            "dataframe.shuffle.compression": "zstd",
            "admin.tick.limit": "3s"
        })

    def _infer_dtypes(self) -> dict:
        """智能类型推断（多格式支持）"""
        path = str(self.settings['path'])

        # 多格式采样逻辑
        if path.endswith('.parquet'):
            sample = dd.read_parquet(path).head(n=1000)
        elif path.endswith('.hdf5'):
            sample = dd.read_hdf(path, key='/data').head(n=1000)
        elif path.endswith('.json'):
            sample = dd.read_json(path, lines=True).head(n=1000)
        else:
            sample = dd.read_csv(path, blocksize="10MB").head(n=1000)

        # 优化类型推断逻辑
        dtypes = {}
        for col in sample.columns:
            col_data = sample[col]
            if pd.api.types.is_numeric_dtype(col_data):
                # 自动降级数值类型
                if col_data.min() >= 0:
                    dtypes[col] = self._optimize_unsigned(col_data)
                else:
                    dtypes[col] = self._optimize_signed(col_data)
            else:
                nunique = col_data.nunique()
                if nunique < min(1000, len(sample) // 10):
                    dtypes[col] = 'category'
                else:
                    dtypes[col] = pd.api.types.infer_dtype(col_data)
        return dtypes

    def _optimize_unsigned(self, series: pd.Series) -> str:
        """无符号数值类型优化"""
        max_val = series.max()
        if max_val <= np.iinfo(np.uint8).max:
            return 'uint8'
        elif max_val <= np.iinfo(np.uint16).max:
            return 'uint16'
        elif max_val <= np.iinfo(np.uint32).max:
            return 'uint32'
        return 'uint64'

    def _optimize_signed(self, series: pd.Series) -> str:
        """有符号数值类型优化"""
        min_val, max_val = series.min(), series.max()
        if (min_val >= np.iinfo(np.int8).min) and (max_val <= np.iinfo(np.int8).max):
            return 'int8'
        elif (min_val >= np.iinfo(np.int16).min) and (max_val <= np.iinfo(np.int16).max):
            return 'int16'
        elif (min_val >= np.iinfo(np.int32).min) and (max_val <= np.iinfo(np.int32).max):
            return 'int32'
        return 'int64'

    def _read_data(self, path: str, dtype: dict) -> dd.DataFrame:
        """统一数据读取接口（消除多分支重复）"""
        read_params = {
            'dtype': dtype,
            'engine': 'pyarrow' if path.endswith('.parquet') else None
        }

        if path.endswith('.parquet'):
            return dd.read_parquet(path, **read_params)
        elif path.endswith('.hdf5'):
            return dd.read_hdf(path, key='/data', mode='r', **read_params)
        elif path.endswith('.json'):
            return dd.read_json(path, lines=True, **read_params)
        else:  # CSV及其他文本格式
            return dd.read_csv(
                path,
                blocksize=self.settings['blocksize'],
                na_values=self.settings['na_value'],
                assume_missing=True,
                on_bad_lines='warn',
                **read_params
            )

    def load_data(self):
        """多格式数据加载（消除重复判断逻辑）"""
        try:
            path = str(self.settings['path'])
            dtype = self.settings.get('dtype', self._infer_dtypes())
            self.ddf = self._read_data(path, dtype)  # 调用统一读取接口
            self._persist_data()
        except Exception as e:
            self.client.restart()
            raise RuntimeError(f"数据加载失败: {str(e)}")

    def _persist_data(self):
        """智能数据持久化策略"""
        if self.ddf is not None:
            self.ddf = self.ddf.persist()
            wait(self.ddf)
            self._optimize_partitions()

    def _optimize_partitions(self):
        """自适应分区优化"""
        ideal_size = 128 * 1024 * 1024  # 128MB per partition
        current_size = self.ddf.memory_usage_per_partition.compute().max()

        if current_size > 2 * ideal_size:
            new_n = max(1, int(self.ddf.memory_usage(deep=True).compute().sum() / ideal_size))
            self.ddf = self.ddf.repartition(npartitions=new_n)
            self.ddf = self.ddf.persist()
            wait(self.ddf)

    def smart_gc(self, force: bool = False) -> bool:
        """增强版内存管理（TB级优化）"""
        cluster_status = self.client.scheduler_info()
        workers_mem = {k: v['metrics']['memory']
                       for k, v in cluster_status['workers'].items()}

        # 使用Dask原生监控接口替代psutil（网页5建议）
        total_used = sum(m['used'] for m in workers_mem.values()) / 1e9
        threshold = 0.75 * self.memory_limit_gb

        if force or total_used > threshold:
            print(f"触发内存回收（Worker使用率：{workers_mem}）")

            # 分阶段清理
            self.client.cancel(self.client.futures)
            self.client.run(lambda: gc.collect(generation=2))

            # 极端内存处理
            if any(p > 0.9 * self.memory_limit_gb for p in workers_mem.values()):
                print("执行滚动重启...")
                self.client.restart()
                self._reload_after_restart()

            return True
        return False

    def _reload_after_restart(self):
        """Worker重启后恢复数据"""
        self.ddf = None
        self.load_data()
        self._persist_data()

    def adaptive_persistence(self, intermediate_df):
        """自适应持久化策略"""
        mem_usage = intermediate_df.memory_usage(deep=True).compute().sum() / 1e9
        if mem_usage > 0.2 * self.memory_limit_gb:
            return intermediate_df.persist()
        return intermediate_df

from PreData2.Outlier import OutlierSetting,DataManager,HandlingOutlier,JudgeMethod,DrawData
import matplotlib.pyplot as plt
class DataProcessingPipeline:
    """数据处理流程接口（整合数据管理、异常处理、分布判断和可视化）"""

    def __init__(self, setting: OutlierSetting, n_workers: int = 4, memory_limit: str = '1GB'):
        """
        初始化数据处理管道
        :param setting: 数据处理配置
        :param n_workers: Dask工作线程数
        :param memory_limit: 内存限制
        """
        self.setting = setting
        self.manager = DataManager(setting, n_workers, memory_limit)
        self.outlier_handler = None
        self.distribution_judge = None
        self.visualizer = None
        self.analysis_results = {}
        self.visualization_results = {}

    def run(self, outlier_method: str = 'iqr', outlier_params: dict = None,
            visualize: bool = True, save_results: bool = True):
        """
        执行完整数据处理流程
        :param outlier_method: 异常值处理方法（iqr/zscore/isolation_forest）
        :param outlier_params: 异常值处理方法参数
        :param visualize: 是否生成可视化
        :param save_results: 是否保存结果
        """
        # 步骤1: 加载数据
        print("=" * 50)
        print("开始加载数据...")
        self.manager.load_data()
        print(f"数据加载完成! 分区数: {self.manager.ddf.npartitions}")
        print("=" * 50 + "\n")

        # 步骤2: 异常值处理
        outlier_params = outlier_params or {}
        print(f"开始异常值处理({outlier_method})...")
        self.outlier_handler = HandlingOutlier(self.manager, original_save=True, visualize=visualize)
        self.outlier_handler.process(method=outlier_method, **outlier_params)
        print("异常值处理完成!")
        print("=" * 50 + "\n")

        # 步骤3: 分布分析
        print("开始分布分析...")
        self.distribution_judge = JudgeMethod(self.manager)
        self._analyze_distributions()
        print("分布分析完成!")
        print("=" * 50 + "\n")

        # 步骤4: 可视化
        if visualize:
            print("生成可视化结果...")
            self.visualizer = DrawData(self.manager)
            self._generate_visualizations()
            print("可视化完成!")
            print("=" * 50 + "\n")

        # 步骤5: 保存结果
        if save_results:
            print("保存处理结果...")
            self._save_results()
            print(f"结果已保存到: {self.setting['save_path']}")
            print("=" * 50 + "\n")

        # 步骤6: 给出处理建议
        print("数据处理建议:")
        self._provide_recommendations()

        # 内存清理
        self.manager.smart_gc(force=True)

    def _analyze_distributions(self):
        """分析所有数值列的分布特征"""
        numeric_cols = self.manager.ddf.select_dtypes(include=np.number).columns.tolist()

        for col in numeric_cols:
            try:
                # 判断是连续型还是离散型
                nunique = self.manager.ddf[col].nunique().compute()
                if nunique > 50:  # 经验阈值
                    result = self.distribution_judge._judge_continuous(col)
                else:
                    sample = self.manager.ddf[col].sample(frac=0.1).compute()
                    result = self.distribution_judge._judge_discrete(sample)

                self.analysis_results[col] = result
                print(f"列 '{col}' 分析结果: {result['distribution']}")

            except Exception as e:
                print(f"列 '{col}' 分布分析失败: {str(e)}")
                self.analysis_results[col] = {'error': str(e)}

    def _generate_visualizations(self):
        """生成关键可视化图表"""
        # 1. 异常值可视化
        if self.outlier_handler and self.setting['save_path']:
            self.outlier_handler.visualize_outliers()

        # 2. 分布直方图
        for col, result in self.analysis_results.items():
            if 'error' not in result:
                try:
                    fig = self.visualizer.plot_distribution(col, 'hist')
                    if self.setting['save_path']:
                        fig.savefig(f"{self.setting['save_path']}_dist_{col}.png")
                    self.visualization_results[f"dist_{col}"] = fig
                except Exception as e:
                    print(f"列 '{col}' 分布图生成失败: {str(e)}")

        # 3. 热力图（带异常值染色）
        try:
            if self.outlier_handler:
                ax = self.visualizer.draw_heatmap_with_outliers(
                    self.outlier_handler,
                    highlight_outliers=True,
                    data_type='cleaned',
                    title="带异常值染色的相关性热力图"
                )
                if self.setting['save_path']:
                    plt.savefig(f"{self.setting['save_path']}_heatmap.png")
                self.visualization_results["heatmap"] = ax
        except Exception as e:
            print(f"热力图生成失败: {str(e)}")

    def _save_results(self):
        """保存处理结果"""
        if not self.setting.get('save_path'):
            return

        # 1. 保存清洗后的数据
        output_path = Path(self.setting['save_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.setting['output_format'] == 'parquet':
            self.manager.ddf.to_parquet(
                f"{self.setting['save_path']}_cleaned.parquet",
                compression=self.setting['compression']
            )
        else:
            self.manager.ddf.to_csv(
                f"{self.setting['save_path']}_cleaned.csv",
                single_file=True,
                compression=self.setting['compression']
            )

        # 2. 保存分析结果
        import json
        with open(f"{self.setting['save_path']}_analysis.json", 'w') as f:
            json.dump(self.analysis_results, f, indent=2)

    def _provide_recommendations(self):
        """根据分布分析结果提供处理建议"""
        for col, result in self.analysis_results.items():
            if 'error' in result:
                print(f"列 '{col}': 分析失败 - {result['error']}")
                continue

            dist_type = result['distribution']
            rec = ""

            if dist_type == 'normal' or dist_type == 'normal-like':
                rec = ("✅ 推荐使用参数检验方法 (如t检验, ANOVA)\n"
                       "✅ 适合使用基于正态分布的机器学习算法 (如线性回归)")
            elif 'skew' in dist_type:
                rec = ("⚠️ 考虑数据转换 (如对数转换)\n"
                       "⚠️ 推荐使用非参数检验方法 (如Mann-Whitney U检验)\n"
                       "✅ 尝试基于树的算法 (如随机森林)")
            elif 'heavy_tail' in dist_type:
                rec = ("⚠️ 考虑鲁棒统计方法 (如中位数替代均值)\n"
                       "⚠️ 使用对异常值不敏感的算法 (如支持向量机)")
            elif dist_type == 'categorical':
                rec = ("✅ 适合使用分类算法 (如逻辑回归, 决策树)\n"
                       "✅ 考虑进行one-hot编码")
            else:
                rec = "ℹ️ 建议进一步探索数据特征"

            print(f"列 '{col}' ({dist_type}) 处理建议:\n{rec}")
            print("-" * 50)

    def get_analysis_results(self) -> dict:
        """获取分布分析结果"""
        return self.analysis_results

    def get_visualizations(self) -> dict:
        """获取生成的可视化图表"""
        return self.visualization_results

    def close(self):
        """关闭管道并释放资源"""
        self.manager.client.close()
        plt.close('all')

# 使用示例
"""
# 配置数据处理参数
setting = OutlierSetting(
    path="large_dataset.csv",
    blocksize="256MB",
    na_value=["NA", "null"],
    dtype={},
    compression="gzip",
    output_format="parquet",
    save_path="processed_data/output"
)

# 创建并运行处理管道
pipeline = DataProcessingPipeline(setting, n_workers=8, memory_limit='4GB')
pipeline.run(
    outlier_method='isolation_forest',
    outlier_params={'contamination': 0.05},
    visualize=True,
    save_results=True
)

# 获取分析结果
results = pipeline.get_analysis_results()
print("年龄列分布类型:", results['age']['distribution'])

# 关闭管道
pipeline.close()
"""