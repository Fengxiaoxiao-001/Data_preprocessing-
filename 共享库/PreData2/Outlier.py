# 异常值
import dask.dataframe as dd
from dask.distributed import Client
from pathlib import Path
from typing import TypedDict, List, Union,Optional,Literal
import numpy as np
import pandas as pd
import gc
import psutil
from scipy import stats
from sklearn.ensemble import IsolationForest
# from sklearn.cluster import DBSCAN
from dask_ml.preprocessing import StandardScaler
from dask_ml.wrappers import ParallelPostFit
from dask.distributed import wait
from scipy.stats import normaltest, anderson
import matplotlib.pyplot as plt  # 添加缺失的导入
import re

class OutlierSetting(TypedDict):
    path: Path
    blocksize: str
    na_value: Union[str, List[str]]
    dtype: dict
    compression: str  # 压缩方式
    output_format: Literal['csv', 'parquet']  # 明确可选值
    save_path: Optional[str]


class DataManager:
    """增强版数据管理器（支持TB级数据处理）"""

    def __init__(self, setting:OutlierSetting , n_workers: int = 4, memory_limit: str = '1GB'):
        self.setting = setting
        self.client = Client(n_workers=n_workers, memory_limit=memory_limit)
        self.ddf = None
        self._current_operations = []
        self.memory_limit_gb = self._parse_memory_limit(memory_limit)

    @staticmethod
    def _parse_memory_limit(limit_str):
        units = {'B': 1e-9, 'KB': 1e-6, 'MB': 1e-3, 'GB': 1, 'TB': 1e3}
        match = re.match(r'^(\d+)([A-Za-z]+)$', limit_str.strip())
        if not match:
            raise ValueError(f"Invalid memory limit format: {limit_str}")
        value, unit = match.groups()
        return float(value) * units[unit.upper()]

    def _infer_dtypes(self):
        """智能类型推断（优化数值类型处理）"""
        path = str(self.setting['path'])
        if path.endswith('.parquet'):
            sample = dd.read_parquet(path).head(n=1000)
        elif path.endswith('.hdf5'):
            sample = dd.read_hdf(path, key='/data').head(n=1000)
        elif path.endswith('.json'):
            sample = dd.read_json(path, lines=True).head(n=1000)
        else:
            sample = dd.read_csv(path, blocksize="10MB").head(n=1000)

        dtypes = {}
        for col in sample.columns:
            col_data = sample[col]
            if pd.api.types.is_numeric_dtype(col_data):
                dtypes[col] = col_data.dtype
            else:
                nunique = col_data.nunique()
                if nunique < min(1000, len(sample) // 10):
                    dtypes[col] = 'category'
                else:
                    dtypes[col] = pd.api.types.infer_dtype(col_data)
        return dtypes

    def load_data(self):
        """多格式数据加载（支持CSV/Parquet/HDF5/JSON）"""
        try:
            path = str(self.setting['path'])
            dtype = self.setting.get('dtype', self._infer_dtypes())

            if path.endswith('.parquet'):
                self.ddf = dd.read_parquet(path, engine='pyarrow', dtype=dtype)
            elif path.endswith('.hdf5'):
                self.ddf = dd.read_hdf(path, key='/data', mode='r', dtype=dtype)
            elif path.endswith('.json'):
                self.ddf = dd.read_json(path, lines=True, dtype=dtype)
            else:
                self.ddf = dd.read_csv(
                    path,
                    blocksize=self.setting['blocksize'],
                    na_values=self.setting['na_value'],
                    dtype=dtype,
                    assume_missing=True,
                    on_bad_lines='warn'
                )
            self._persist_data()
        except Exception as e:
            self.client.restart()
            raise RuntimeError(f"数据加载失败: {str(e)}")

    def _persist_data(self):
        """持久化数据到集群内存"""
        if self.ddf is not None:
            self.ddf = self.ddf.persist()
            wait(self.ddf)

    def smart_gc(self, force=False):
        """智能内存回收（支持TB级数据）"""

        workers_mem = self.client.run(lambda: psutil.Process().memory_info().rss / (1024 ** 3))
        threshold = 0.75 * self.memory_limit_gb
        if force or any(p > threshold for p in workers_mem.values()):
            self.client.cancel(self.client.futures)
            self.client.run(lambda: gc.collect(generation=2))
            if any(p > 90 for p in workers_mem.values()):
                self.client.restart()
                self.ddf = None  # 清除旧引用
                self.load_data()  # 重启后重新加载数据
            return True
        return False

from Visualization.rectangle import DataVisualizer
class HandlingOutlier:
    """异常值处理引擎（优化算法实现）"""

    def __init__(self, manager: DataManager,original_save:bool=False,visualize:bool=False):
        self.manager = manager
        self.original_save = original_save
        self._original_ddf = None  # 新增：存储原始数据副本
        self._cleaned_ddf = None  # 新增：存储清洗后数据
        self._outlier_mask = None  # 新增：存储异常值掩码
        # 新增：可视化引擎
        self.visualize = visualize
        self.visualizer = None

    def process(self, method: str = 'iqr', columns: List[str] = None, **params):
        if self.original_save:
            # 在处理前保存原始数据副本
            self._original_ddf = self.manager.ddf.copy()

        if columns is None:
            columns = self.manager.ddf.select_dtypes(include=np.number).columns.tolist()

        # 处理前可视化
        if self.visualize:
            self.visualize_pre_processing(columns)

        if method == 'iqr':
            self._iqr_method(columns, **params)
        elif method == 'zscore':
            self._zscore_method(columns, **params)
        elif method == 'isolation_forest':
            self._isolation_forest_method(columns, **params)
        else:
            raise ValueError(f"不支持的异常检测方法: {method}")

        # 处理后可视化
        if self.visualize:
            self.visualize_post_processing(columns)

        if self.original_save:
            # 处理后保存清洗后数据和掩码
            self._cleaned_ddf = self.manager.ddf
            self._generate_outlier_mask()


    def _iqr_method(self, columns: List[str], range: float = 1.5):
        """优化后的IQR方法"""
        ddf = self.manager.ddf
        q = ddf[columns].quantile([0.25, 0.75], method='tdigest').compute()
        iqr = q.loc[0.75] - q.loc[0.25]
        outlier_cond = None

        for col in columns:
            lower = q.loc[0.25][col] - range * iqr[col]
            upper = q.loc[0.75][col] + range * iqr[col]
            col_cond = (ddf[col] < lower) | (ddf[col] > upper)
            outlier_cond = col_cond if outlier_cond is None else outlier_cond | col_cond

        self.manager.ddf = ddf[~outlier_cond]
        self.manager._persist_data()

    def _zscore_method(self, columns: List[str], threshold: float = 3):
        """优化Z-score方法"""
        scaler = StandardScaler()
        scaler.fit(self.manager.ddf[columns])  # 分布式拟合
        scaled = scaler.transform(self.manager.ddf[columns])  # 分布式转换
        mask = (abs(scaled) < threshold).all(axis=1)
        self.manager.ddf = self.manager.ddf[mask].persist()

    def _isolation_forest_method(self, columns: List[str], contamination: float = 0.1):
        """添加Isolation Forest方法"""
        sample = self.manager.ddf[columns].sample(n=10000).compute()
        model = IsolationForest(contamination=contamination)
        model.fit(sample)

        parallel_model = ParallelPostFit(model)
        preds = parallel_model.predict(self.manager.ddf[columns])
        self.manager.ddf = self.manager.ddf[preds == 1]
        self.manager._persist_data()

    # 适配热力图
    def get_clean_data(self) -> dd.DataFrame:
        """获取清洗后的数据（独立副本）"""
        return self._cleaned_ddf.copy() if self._cleaned_ddf is not None else self.manager.ddf

    def get_outlier_flags(self) -> dd.DataFrame:
        """获取异常值标记矩阵"""
        if self._outlier_mask is None:
            # 返回全False的掩码
            sample = self.manager.ddf.head(1)
            mask = sample.map_partitions(lambda df: pd.DataFrame(False, index=df.index, columns=df.columns))
            return mask
        return self._outlier_mask

    def _generate_outlier_mask(self):
        """生成与原始数据对齐的异常值掩码"""
        if self._original_ddf is None or self._cleaned_ddf is None:
            return

        # 计算原始数据和清洗后数据的差异
        original_computed = self._original_ddf.compute()
        cleaned_computed = self._cleaned_ddf.compute()

        # 对齐索引（关键修复）
        aligned_original, aligned_cleaned = original_computed.align(
            cleaned_computed,
            join='left',  # 保留所有原始行
            fill_value=None
        )

        # 创建布尔掩码（True表示异常值）
        mask = aligned_cleaned.isna()  # 清洗后数据中缺失的位置即为被移除的异常值
        self._outlier_mask = dd.from_pandas(
            mask,
            npartitions=self._original_ddf.npartitions
        )

    # 柱形图/条形图（类别对比）和直方图（数据分布）
    # 新增：初始化可视化引擎（按需创建）
    def _init_visualizer(self):
        if self.visualizer is None:
            self.visualizer = DataVisualizer(style="whitegrid", palette="pastel", figsize=(12, 8))
        return self.visualizer

    # 新增：处理前可视化方法
    def visualize_pre_processing(self, columns: List[str]):
        """处理前数据可视化"""
        if not columns:
            return

        print("\n" + "=" * 50)
        print("异常值处理前数据分布可视化")
        print("=" * 50)

        visualizer = self._init_visualizer()

        # 对每列生成可视化
        for col in columns:
            try:
                # 获取数据样本
                sample = self.manager.ddf[col].sample(frac=0.1).compute() if self.manager.ddf.npartitions > 1 \
                    else self.manager.ddf[col].compute()

                # 创建子图
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # 舍弃状态
                '''
                # 箱线图 - 使用DataVisualizer
                visualizer.bar_plot(
                    data=pd.DataFrame({col: sample}),
                    x=col,
                    y=None,
                    title=f'【{col}】处理前箱线图',
                    xlabel='值范围',
                    orientation="vertical",
                    ax=axes[0]
                )
                '''

                # 直方图 - 使用DataVisualizer
                visualizer.hist_plot(
                    data=pd.DataFrame({col: sample}),
                    column=col,
                    title=f'【{col}】处理前分布',
                    xlabel='值',
                    ylabel='频数',
                    ax=axes[1]
                )

                plt.tight_layout()

                # 保存或显示
                if self.manager.setting.get('save_path'):
                    plt.savefig(f"{self.manager.setting['save_path']}_pre_{col}.png", dpi=300)
                plt.show()

            except Exception as e:
                print(f"列 {col} 可视化失败: {str(e)}")

    # 新增：处理后可视化方法
    def visualize_post_processing(self, columns: List[str]):
        """处理后数据可视化"""
        if not columns:
            return

        print("\n" + "=" * 50)
        print("异常值处理后数据分布可视化")
        print("=" * 50)

        visualizer = self._init_visualizer()

        # 对每列生成可视化
        for col in columns:
            try:
                # 获取数据样本
                sample = self.manager.ddf[col].sample(frac=0.1).compute() if self.manager.ddf.npartitions > 1 \
                    else self.manager.ddf[col].compute()

                # 创建子图
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # 箱线图 - 使用DataVisualizer
                visualizer.bar_plot(
                    data=pd.DataFrame({col: sample}),
                    x=col,
                    y=None,
                    title=f'【{col}】处理后箱线图',
                    xlabel='值范围',
                    orientation="vertical",
                    ax=axes[0]
                )

                # 直方图 - 使用DataVisualizer
                visualizer.hist_plot(
                    data=pd.DataFrame({col: sample}),
                    column=col,
                    title=f'【{col}】处理后分布',
                    xlabel='值',
                    ylabel='频数',
                    ax=axes[1]
                )

                plt.tight_layout()

                # 保存或显示
                if self.manager.setting.get('save_path'):
                    plt.savefig(f"{self.manager.setting['save_path']}_post_{col}.png", dpi=300)
                plt.show()

            except Exception as e:
                print(f"列 {col} 可视化失败: {str(e)}")

    # 新增：异常值标记可视化方法
    def visualize_outliers(self, columns: List[str] = None):
        """异常值标记可视化"""
        import matplotlib.patches as mpatches
        if not self.original_save or self._outlier_mask is None:
            print("无法可视化异常值：缺少原始数据或异常值掩码")
            return

        if columns is None:
            columns = self._outlier_mask.columns.tolist()

        print("\n" + "=" * 50)
        print("异常值分布可视化")
        print("=" * 50)

        for col in columns:
            try:
                # 获取原始数据
                original_data = self._original_ddf[col].compute()

                # 获取异常值掩码
                outlier_mask = self._outlier_mask[col].compute()

                # 创建可视化
                plt.figure(figsize=(12, 6))

                # 绘制原始数据点
                plt.scatter(
                    x=range(len(original_data)),
                    y=original_data,
                    c=outlier_mask.map({True: 'red', False: 'blue'}),
                    alpha=0.6,
                    s=20
                )

                # 添加说明
                plt.title(f'【{col}】异常值分布（红色为异常值）')
                plt.xlabel('数据索引')
                plt.ylabel('值')

                # 添加图例
                normal_patch = mpatches.Patch(color='blue', label='正常值')
                outlier_patch = mpatches.Patch(color='red', label='异常值')
                plt.legend(handles=[normal_patch, outlier_patch])

                plt.grid(alpha=0.3)

                # 保存或显示
                if self.manager.setting.get('save_path'):
                    plt.savefig(f"{self.manager.setting['save_path']}_outliers_{col}.png", dpi=300)
                plt.show()

            except Exception as e:
                print(f"列 {col} 异常值可视化失败: {str(e)}")

from Visualization.rectangle import HeatmapGenerator
class DrawData:
    """大数据可视化引擎"""
    def __init__(self, manager:DataManager):
        self.manager = manager
        self.heatmap_generator = None  # 热力图生成器实例

    def plot_distribution(self, column: str, plot_type: str = 'hist', **kwargs):
        """
        核心绘图方法
        :param column: 需要分析的列名
        :param plot_type: 支持 hist/box/qq 等
        :param kwargs: 各图表专用参数
        """
        ddf = self.manager.ddf

        if plot_type == 'hist':
            return self._plot_histogram(ddf, column, **kwargs)
        elif plot_type == 'box':
            return self._plot_boxplot(ddf, column)
        elif plot_type == 'qq':
            return self._plot_qq(ddf, column, **kwargs)
        else:
            raise ValueError(f"不支持的图表类型: {plot_type}")

    def _plot_histogram(self,ddf: dd.DataFrame, column: str, bins: int = 100):
        """分布式直方图绘制（优化内存使用）"""
        # 计算全局最小最大值保证分箱一致性
        min_val = ddf[column].min().compute()
        max_val = ddf[column].max().compute()
        if min_val == max_val:
            min_val -= 0.5
            max_val += 0.5
        bin_edges = np.linspace(min_val, max_val, bins + 1)

        def compute_hist(s):
            return pd.cut(s, bins=bin_edges).value_counts()

        hist = ddf[column].map_partitions(
            compute_hist,
            meta=pd.Series(dtype='float64'))
        hist = hist.groupby(hist.index).sum().compute()
        hist.sort_index().plot.bar()
        plt.title(f'Distribution of {column}')
        if self.manager.setting['save_path']:
            plt.savefig(self.manager.setting['save_path'])
            plt.show()
        else:
            plt.show()

    def _plot_boxplot(self,ddf: dd.DataFrame, column: str):
        """分布式箱线图计算（优化分位数计算）"""
        stats = ddf[column].describe(percentiles=[.25, .5, .75]).compute()
        plt.figure()
        plt.boxplot(ddf[column].compute(), vert=True)
        plt.title(f'Boxplot of {column}')
        if self.manager.setting['save_path']:
            plt.savefig(self.manager.setting['save_path'])
            plt.show()
        else:
            plt.show()
        return {
            'min': stats['min'],
            'q1': stats['25%'],
            'median': stats['50%'],
            'q3': stats['75%'],
            'max': stats['max']
        }

    def _plot_qq(self,ddf: dd.DataFrame, column: str, sample_size: int = 5000):
        """Q-Q图（优化抽样方法）"""
        sample = ddf[column].sample(n=sample_size).compute()
        stats.probplot(sample, plot=plt)
        plt.title('Q-Q Plot')
        if self.manager.setting['save_path']:
            plt.savefig(self.manager.setting['save_path'])
            plt.show()
        else:
            plt.show()

    def draw_heatmap_with_outliers(self,
                                   outlier_handler: HandlingOutlier,
                                   highlight_outliers: bool = False,
                                   **kwargs) -> plt.Axes:
        """
        集成异常值染色的热力图生成[7](@ref)
        :param outlier_handler: 异常值处理器实例
        :param kwargs: 透传给create_heatmap的参数
        """
        # 生成基础热力图
        hm = self._create_heatmap(outlier_handler,**kwargs)

        # 自动判断是否需要异常值染色
        should_highlight = (
                highlight_outliers and
                outlier_handler is not None and
                outlier_handler.get_outlier_flags() is not None
        )

        if should_highlight:
            # 获取异常值掩膜（需在HandlingOutlier中新增方法）
            outlier_mask = self._get_outlier_mask(outlier_handler)

            # 确保掩码与热力图数据对齐
            common_cols = outlier_mask.columns.intersection(hm.data.columns)
            aligned_mask = outlier_mask[common_cols]

            # 叠加异常值染色层
            hm.add_outlier_layer(aligned_mask)

        # 绘制并返回
        if self.manager.setting['save_path']:
            hm.save(self.manager.setting['save_path'])
        return hm.draw()

    def _create_heatmap(self,
                        handling_outlier:HandlingOutlier,
                       data_type: Literal['raw', 'cleaned'] = 'raw',            # 原始数据（raw）或清洗后数据（cleaned）
                       columns: List[str] = None,                               # columns: 指定分析的列名（默认全量）
                       title: str = "Feature Correlation Heatmap",
                       color_map: str = "icefire",
                       ) -> HeatmapGenerator:
        """
        创建热力图生成器实例[7](@ref)
        :param data_type: 使用原始数据(raw)或清洗后数据(cleaned)
        :param columns: 指定分析的列名列表
        :param title: 热力图标题
        :param color_map: 配色方案（viridis/icefire/coolwarm）
        """
        # 获取数据源                             \ 是 Python 中的续行符
        data_source = self.manager.ddf if data_type == 'raw' \
            else handling_outlier.get_clean_data()

        # 计算相关性矩阵
        corr_data = data_source[columns].corr().compute() if columns \
            else data_source.corr().compute()
        '''
        指定 columns 时，仅计算选定列的相关性
        未指定时，计算全量数据相关性
        '''
        # 实例化热力图生成器
        self.heatmap_generator = HeatmapGenerator(
            corr_data,
            title=title,
            color_map=color_map,
            annot=True  # 大数据时，一般annot=False 表示关闭热力图的数值标注，避免因数据量过大导致内存溢出 大数据时
        )
        return self.heatmap_generator

    def _get_outlier_mask(self, handler: HandlingOutlier) -> pd.DataFrame:
        """生成索引对齐的布尔掩膜矩阵"""
        # 获取原始数据和清洗后数据
        original = self.manager.ddf.compute()
        cleaned = handler.get_clean_data().compute()

        # 显式重建索引对齐
        # aligned_original = original.reindex(index=cleaned.index, columns=cleaned.columns)   # 若 cleaned 的索引/列在 original 中不存在，aligned_original 会引入 NaN
        # aligned_cleaned = cleaned.reindex(index=original.index, columns=original.columns)
        # 使用align()方法一步对齐（避免两次reindex）
        aligned_original, aligned_cleaned = original.align(cleaned, join='outer', fill_value=np.nan)

        # 生成布尔矩阵（使用不同的变量名避免混淆
        outlier_mask = (aligned_original.fillna(0) != aligned_cleaned.fillna(0))    #  通过填充缺失值为 0 后比较生成布尔掩膜
        # 将 NaN 填充为 0 后，差异位置被标记为 True

        # 正确使用 fillna 处理缺失值（在 DataFrame 上操作）
        return outlier_mask.fillna(False)

class JudgeMethod:
    """分布类型判断引擎（优化大样本处理）"""
    def __init__(self,manager:DataManager):
        self.manager = manager

    def _judge_continuous(self, column: str) -> dict:
        """综合改进的连续型分布判断"""
        # 使用近似计算加速
        total = self.manager.ddf[column].count().compute()

        # 并行计算偏度和峰度
        skew_kurt = self.manager.ddf[column].agg(['skew', 'kurtosis'],
                                                 split_every=10).compute()
        skewness = skew_kurt['skew']
        kurtosis = skew_kurt['kurtosis']

        # 改进的分布类型判断
        distribution_type = self._determine_dist_type(skewness, kurtosis)

        # 分层抽样策略（优化尾部数据捕捉）
        sample_size = self._get_sample_size(total)
        sample = self._stratified_sampling(column, sample_size)

        # 流式处理（TB级数据专用）
        if total > 1e9:
            self._streaming_statistics(column)

        # 多方法正态检验
        norm_info = self._run_normality_tests(sample)

        # 综合判断逻辑
        is_normal = self._judge_normality(norm_info)

        return {
            'distribution': 'normal' if is_normal else distribution_type,
            'skew_type': self._classify_skew(skewness),
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'norm_tests': norm_info,
            'sample_size': sample_size
        }

    @staticmethod
    def _get_sample_size( total: int) -> int:
        """动态样本量策略"""
        if total > 1e8:
            return 100_000  # 大数据抽样
        elif total > 1e6:
            return 10_000
        return min(5000, total)

    def _stratified_sampling(self, column: str, sample_size: int) -> pd.Series:
        """分层抽样（保证尾部数据）"""
        # 使用分位数分层
        ddf = self.manager.ddf
        quantiles = ddf[column].quantile([0.25, 0.5, 0.75], method='tdigest').compute()

        # 定义分层区间
        bins = [-np.inf, quantiles[0], quantiles[1], quantiles[2], np.inf]
        labels = ['Q1', 'Q2', 'Q3', 'Q4']

        ddf = ddf.assign(strata=lambda df: pd.cut(df[column], bins=bins, labels=labels))
        strata_counts = ddf['strata'].value_counts().compute()
        total = strata_counts.sum()
        sample_per_stratum = {s: int(sample_size * c / total) for s, c in strata_counts.items()}
        samples = []
        for stratum, n in sample_per_stratum.items():
            if n == 0:
                continue
            sampled = ddf[ddf['strata'] == stratum].sample(n=n, random_state=42)
            samples.append(sampled)
        combined = dd.concat(samples).compute()[column]
        return combined

    def _streaming_statistics(self, column: str):
        """TB级数据流式统计（优化增量计算）"""
        if self.manager.ddf is not None:
            # 使用增量统计方法
            mean = self.manager.ddf[column].mean()
            var = self.manager.ddf[column].var()
            count = self.manager.ddf[column].count()

            # 并行计算三个统计量
            stats = dd.compute(mean, var, count)

            stream_mean = stats[0]
            stream_std = np.sqrt(stats[1])
            total_count = stats[2]

            print(f"实时统计量[增量] - {column}: μ={stream_mean:.2f}, σ={stream_std:.2f}, n={total_count:,}")

    @staticmethod
    def _run_normality_tests( sample: pd.Series) -> dict:
        """综合正态性检验"""
        norm_info = {}

        # D'Agostino's K² test（适用于n>50）
        if len(sample) >= 50:
            _, p_normal = normaltest(sample)
            norm_info['dagostino_p'] = p_normal

        # Anderson-Darling（大样本优化）
        anderson_result = anderson(sample, dist='norm')
        norm_info.update({
            'anderson_stat': anderson_result.statistic,
            'critical_values': anderson_result.critical_values
        })

        # Shapiro-Wilk（小样本参考）
        if 3 <= len(sample) <= 5000:
            _, p_shapiro = stats.shapiro(sample)
            norm_info['shapiro_p'] = p_shapiro

        return norm_info

    @staticmethod
    def _judge_normality(norm_info: dict) -> bool:
        """综合判断正态性"""
        # Anderson-Darling在5%显著性水平
        anderson_pass = norm_info['anderson_stat'] < norm_info['critical_values'][2]

        # D'Agostino检验
        dagostino_pass = norm_info.get('dagostino_p', 0.0) > 0.05

        # 综合判断逻辑
        return anderson_pass and dagostino_pass

    @staticmethod
    def _determine_dist_type(skewness: float, kurtosis: float) -> str:
        """基于矩量的分布类型判断（增强版）"""
        abs_skew = abs(skewness)
        abs_kurt = abs(kurtosis - 3)  # 相对于正态分布的峰度

        if abs_skew < 0.5 and abs_kurt < 1:
            return 'normal-like'
        elif skewness > 1 and kurtosis > 3:
            return 'right_skew_heavy_tail'
        elif skewness < -1 and kurtosis > 3:
            return 'left_skew_heavy_tail'
        elif abs_skew < 1 and abs_kurt > 3:
            return 'non-normal_light_tail'
        else:
            return 'non-normal_complex'

    @staticmethod
    def _classify_skew(s: float) -> str:
        """动态偏态分类（结合数据规模）"""
        abs_s = abs(s)
        if abs_s < 0.3:
            return 'symmetric'
        elif abs_s < 0.7:
            return 'mild'
        elif abs_s < 1.5:
            return 'moderate'
        elif abs_s < 3:
            return 'high'
        else:
            return 'extreme'

    @staticmethod
    def _judge_discrete(series: pd.Series) -> dict:
        """离散型分布判断"""
        nunique = series.nunique()
        return {
            'distribution': 'categorical' if nunique < 10 else 'multimodal',
            'suggested_model': 'Poisson' if nunique > 1 else 'Bernoulli'
        }


# 使用示例
"""
if __name__ == "__main__":

    settings = OutlierSetting(
        path= 'large_dataset.csv',
        blocksize= '256MB',
        na_value= ['NA', 'missing'],
        dtype= {'price': 'float64', 'quantity': 'int32'},
        compression= 'gzip',
        output_format= 'parquet'
    )

    # 初始化数据管理器
    manager = DataManager(settings)
    manager.load_data()

    # 可视化分析
    drawer = DrawData(manager)
    drawer.plot_distribution(column='price', plot_type='box')

    # 分布判断
    judge = JudgeMethod(manager)
    dist_report = judge.judge_distribution('price')
    print(f"分布判断结果：{dist_report}")

    # 异常值处理
    outlier_handler = HandlingOutlier(manager)
    outlier_handler.process(method='iqr', columns=['price', 'quantity'])

    # 资源清理
    manager.smart_gc(force=True)
"""