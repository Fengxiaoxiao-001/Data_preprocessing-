# 异常值
import dask.dataframe as dd
from dask.distributed import Client
from pathlib import Path
from typing import TypedDict, List, Union, Optional, Literal
import numpy as np
import pandas as pd
import gc
import psutil
from scipy import stats
from sklearn.ensemble import IsolationForest
from dask_ml.preprocessing import StandardScaler
from dask_ml.wrappers import ParallelPostFit
from dask.distributed import wait
from scipy.stats import normaltest, anderson
import matplotlib.pyplot as plt
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

    def __init__(self, setting: OutlierSetting, n_workers: int = 4, memory_limit: str = '1GB'):
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


class HandlingOutlier:
    """异常值处理引擎（优化算法实现）"""

    def __init__(self, manager: DataManager, original_save: bool = False):
        self.manager = manager
        self.original_save = original_save
        self._original_ddf = None  # 新增：存储原始数据副本
        self._cleaned_ddf = None  # 新增：存储清洗后数据
        self._outlier_mask = None  # 新增：存储异常值掩码

    def process(self, method: str = 'iqr', columns: List[str] = None, **params):
        if self.original_save:
            # 在处理前保存原始数据副本
            self._original_ddf = self.manager.ddf.copy()

        if columns is None:
            columns = self.manager.ddf.select_dtypes(include=np.number).columns.tolist()

        if method == 'iqr':
            self._iqr_method(columns, **params)
        elif method == 'zscore':
            self._zscore_method(columns, **params)
        elif method == 'isolation_forest':
            self._isolation_forest_method(columns, **params)
        elif method == 'mad':
            self._mad_method(columns, **params)
        elif method == 'lof':
            self._lof_method(columns, **params)
        elif method == 'dbscan':
            self._dbscan_method(columns, **params)
        elif method == 'ked':
            self._kde_method(columns, **params)
        elif method == 'grubbs':
            self._grubbs_metho(columns, **params)
        elif method == 'tukey':
            self._tukey_method(columns, **params)
        elif method == 'discrete_dist':
            self._discrete_dist_method(columns, **params)
        elif method == 'weibull':
            self._weibull_method(columns, **params)
        elif method == 'lognormal':
            self._lognormal_method(columns, **params)
        else:
            raise ValueError(f"不支持的异常检测方法: {method}")

        if self.original_save:
            # 处理后保存清洗后数据和掩码
            self._cleaned_ddf = self.manager.ddf
            self._generate_outlier_mask()

    # 异常值处理方法
    def _iqr_method(self, columns: List[str], base_range: float = 1.5, target_outlier_rate: float = 0.01):
        """优化后的IQR方法（箱线图法）
        适用于数值型数据，尤其适合非正态分布（如偏态分布、多峰分布）
        """
        ddf = self.manager.ddf
        q = self._optimized_quantile(columns, [0.25, 0.75])
        iqr = q.loc[0.75] - q.loc[0.25]
        outlier_cond = None

        if not hasattr(self, '_historical_outlier_rates'):
            self._historical_outlier_rates = {col: target_outlier_rate for col in columns}

        for col in columns:
            col_data = ddf[col]

            dynamic_range = self._calc_dynamic_range(
                col_data=col_data,
                base_range=base_range,
                historical_rate=self._historical_outlier_rates.get(col, target_outlier_rate),
                target_rate=target_outlier_rate
            )

            lower = q.loc[0.25][col] - dynamic_range * iqr[col]
            upper = q.loc[0.75][col] + dynamic_range * iqr[col]

            outlier_mask = (col_data < lower) | (col_data > upper)
            current_outlier_rate = outlier_mask.mean().compute()
            self._historical_outlier_rates[col] = current_outlier_rate  # 更新历史记录

            col_cond = outlier_mask
            outlier_cond = col_cond if outlier_cond is None else outlier_cond | col_cond

        self.manager.ddf = ddf[~outlier_cond]
        self.manager._persist_data()

    def _zscore_method(self, columns: List[str], threshold: float = 3):
        """优化Z-score方法
        适用于数值型数据，要求数据近似正态分布（否则误判率高）"""
        scaler = StandardScaler()
        scaler.fit(self.manager.ddf[columns])
        scaled = scaler.transform(self.manager.ddf[columns])

        mask = None

        dyn_thresholds = {}
        for col in columns:

            dyn_thresholds[col] = self._calc_dynamic_z_threshold(
                col_data=self.manager.ddf[col],
                base_threshold=threshold
            )

            for idx, col in enumerate(columns):
                col_mask = (abs(scaled[:, idx]) < dyn_thresholds[col])
                mask = col_mask if mask is None else (mask & col_mask)

        self.manager.ddf = self.manager.ddf[mask].persist()

    def _isolation_forest_method(self, columns: List[str], contamination: float = 0.1, auto_adjust: bool = True):
        """Isolation Forest方法
        这是一种基于机器学习的无监督方法
        适用数据类型：
        <高维数值>型数据和无需假设数据分布，适合复杂非线性关系数据"""
        sample = self.manager.ddf[columns].sample(n=10000).compute()


        if auto_adjust:
            prelim_model = IsolationForest(
                contamination=contamination,  # 临时值
                n_estimators=50,  # 加速初步训练
                max_samples=512  # 减小计算量
            )
            prelim_model.fit(sample)

            contamination = self._score_based_contamination(prelim_model, columns)
            contamination = min(0.3, max(0.01, contamination))  # 限制范围[1%,30%]

        model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            max_samples="auto"
        )
        model.fit(sample)

        # 分布式预测
        parallel_model = ParallelPostFit(model)
        preds = parallel_model.predict(self.manager.ddf[columns])

        self.manager.ddf = self.manager.ddf[preds == 1]
        self.manager._persist_data()

    def _mad_method(self, columns: List[str], threshold: float = 3, auto_adjust: bool = True):
        """MAD（绝对中位差法）
        尖峰厚尾分布（金融数据）
        适用场景：金融量化、股票数据（尖峰厚尾分布）"""

        med = self.manager.ddf[columns].median()
        abs_dev = (self.manager.ddf[columns] - med).abs()
        mad = (self.manager.ddf[columns] - med).abs().median()
        mad_scaled = 1.4826 * mad

        skews = self.manager.ddf[columns].skew().compute() if auto_adjust else {}

        mask = self.manager.ddf.map_partitions(
            lambda df: pd.Series(True, index=df.index),
            meta=('mask', bool)
        )

        # 列向量化处理
        for col in columns:
            dyn_threshold = threshold
            if auto_adjust:

                # 偏度自适应阈值（提取标量值）
                skew_val = skews[col] if isinstance(skews[col], float) else skews[col][0]
                dyn_threshold *= (1 + min(1.0, abs(skew_val) / 2))
                dyn_threshold = min(dyn_threshold, 5.0)  # 阈值上限保护

            # 列级掩码计算
            col_mask = (abs_dev[col] <= dyn_threshold * mad_scaled[col])
            mask = mask & col_mask

            self.manager.ddf = self.manager.ddf[mask]

    def _lof_method(self, columns: List[str], n_neighbors: int = 20):
        """LOF（局部异常因子）
        局部密度变化（空间数据）
        适用场景：空间数据、聚类中的局部异常"""
        from sklearn.neighbors import LocalOutlierFactor

        sample = self.manager.ddf[columns].sample(n=10000).compute()

        model = LocalOutlierFactor(n_neighbors=n_neighbors,
                                   novelty=True)
        # 模型训练
        model.fit(sample)
        # 全量数据预测
        preds = model.predict(self.manager.ddf[columns].compute())

        self.manager.ddf = self.manager.ddf[preds == 1]


    def _dbscan_method(
            self,
            columns: List[str],
            eps: float = None,  # 支持自动计算
            auto_eps_method: str = "kdist",  # 可选 "kdist" 或 "sampling"
            k: int = 10,
            sample_size: int = 10000
    ):
        """DBSCAN（基于密度聚类）
               聚类异常（地理信息）
               适用场景：地理信息、传感器网络"""
        from sklearn.cluster import DBSCAN

        # 自动计算eps（若未指定）
        if eps is None:
            if auto_eps_method == "kdist":
                eps = self._auto_eps_kdist(columns, k)
            else:
                eps = self._auto_eps_sampling(columns, sample_size, k)
            print(f"Auto-selected eps: {eps:.4f}")  # 输出调试信息

        # 执行聚类
        sample = self.manager.ddf[columns].compute()
        labels = DBSCAN(eps=eps).fit_predict(sample)
        mask = labels != -1
        self.manager.ddf = self.manager.ddf[mask]

    def _kde_method(self, columns: List[str], quantile: float = 0.05):
        """核密度估计法
        适用分布：U型/J型/多峰分布等复杂分布
        原理：通过概率密度函数识别低密度区域
        """
        from sklearn.neighbors import KernelDensity
        import dask.array as da

        sample = self.manager.ddf[columns].sample(frac=0.1).compute().values
        kde = KernelDensity().fit(sample)

        # 计算采样数据的概率密度
        log_probs = kde.score_samples(sample)
        probs = np.exp(log_probs)

        threshold = np.quantile(probs, quantile)

        dask_array = self.manager.ddf[columns].to_dask_array(lengths=True)

        # 定义一个函数处理每个块
        def predict_chunk(
                chunk):

            log_probs = kde.score_samples(chunk)
            return np.exp(log_probs) > threshold

        # 使用 map_blocks 并行处理
        mask_array = da.map_blocks(predict_chunk, dask_array, dtype=bool)

        mask_series = dd.from_dask_array(mask_array, columns=['mask']).squeeze()


        # 使用 mask_series 进行过滤
        self.manager.ddf = self.manager.ddf[mask_series]

    def _grubbs_metho(self, columns: List[str], alpha: float = 0.05,
                      sample_size: int = 10000, iter_max: int = 5):
        """Grubbs检验法
        适用分布：正态/近正态分布（严格正态分布）
        原理：基于标准差的极端值检测
        α 表示显著性水平（检出水平）
        α = 0.05 对应 95% 置信度（检出水平）
        α = 0.01 对应 99% 置信度（剔除水平）
        """
        from dask.distributed import Client
        import numpy as np
        from scipy import stats

        with Client(n_workers=8, threads_per_worker=2, memory_limit='2GB') as client:  # 上下文管理器自动关闭

            for col in columns:
                # 步骤1：分布式计算全局统计量（避免全量加载）
                global_mean = self.manager.ddf[col].mean().compute()
                global_std = self.manager.ddf[col].std().compute()

                # 步骤2：分层采样（保留分布特征）
                sampled = self.manager.ddf[col].sample(frac=sample_size / len(self.manager.ddf),
                                                       random_state=42).compute()

                # 步骤3：在样本上迭代检测异常值
                outlier_indices = set()
                for _ in range(iter_max):
                    if len(sampled) < 3: break

                    deviations = np.abs(sampled - global_mean)
                    idx = deviations.idxmax()
                    g = deviations.max() / global_std

                    # 计算临界值（使用原始数据量n）
                    n_original = len(self.manager.ddf)
                    t = stats.t.ppf(1 - alpha / (2 * n_original), n_original - 2)
                    threshold = (n_original - 1) / np.sqrt(n_original) * np.sqrt(t ** 2 / (n_original - 2 + t ** 2))

                    if g > threshold:
                        outlier_indices.add(idx)
                        sampled = sampled.drop(idx)  # 更新样本
                    else:
                        break

                # 步骤4：分布式过滤异常值
                if outlier_indices:
                    mask = ~self.manager.ddf.index.isin(list(outlier_indices))
                    self.manager.ddf = self.manager.ddf[mask]  # 仅保留 mask 为 True 的行（即非异常值）

                # 过滤后立即释放采样数据
                del sampled  # 显式释放内存

            # 分布式对象主动释放
            client.run(lambda: gc.collect())  # 强制所有Worker垃圾回收

        # 退出时自动调用 client.close() + 隐式 shutdown

    def _tukey_method(self, columns: List[str], k: float = 1.5, auto_adjust: bool = True):
        """Tukey's Fences
        适用分布：偏态/指数分布
        原理：改进IQR的偏态适应方法
        金融风控：需严格检测（k=1.0~1.5）
        生态监测：需保留自然波动（k=2.0~3.0）
        """
        # 使用优化分位数计算获取Q1和Q3
        q = self._optimized_quantile(columns, quantiles=[0.25, 0.75])
        # 计算IQR（四分位距）
        iqr = q.loc[0.75] - q.loc[0.25]

        for col in columns:
            # 动态计算k值
            if auto_adjust:
                from scipy.stats import skew
                col_skew = skew(self.manager.ddf[col].compute())
                k_adj = k * (1 + abs(col_skew) / 3)
            else:
                k_adj = k

            # 计算异常值边界
            lower = q.loc[0.25][col] - k_adj * iqr[col]
            upper = q.loc[0.75][col] + k_adj * iqr[col]

            # 创建过滤掩码
            mask = (self.manager.ddf[col] >= lower) & (self.manager.ddf[col] <= upper)
            self.manager.ddf = self.manager.ddf[mask]

    def _discrete_dist_method(self, columns: List[str], dist_type: str,
                              threshold: float = 3, auto_adjust: bool = True, n_trials: Union[int, dict] = None):
        """
        离散分布异常检测（泊松/二项分布专用）
        参数：
        dist_type: 'poisson' 或 'binomial'
        n_trials: 二项分布时需提供试验次数（可列名字典或统一值）
        """
        ddf = self.manager.ddf
        outlier_cond = None

        for col in columns:
            col_data = ddf[col]

            # 单次归约获取多统计量
            stats = col_data.reduction(
                lambda chunk: (chunk.sum(), chunk.count(), (chunk ** 2).sum()),
                lambda x, y: (x[0] + y[0], x[1] + y[1], x[2] + y[2]),
                meta=[('sum', 'f8'), ('count', 'i8'), ('sum_sq', 'f8')]
            ).compute()

            total_sum, total_count, sum_sq = stats
            mean = total_sum / total_count
            variance = (sum_sq - total_sum ** 2 / total_count) / (total_count - 1)
            std = np.sqrt(variance)

            # 动态阈值计算
            dyn_threshold = self._calc_dynamic_threshold(col_data, threshold) if auto_adjust else threshold

            if dist_type == 'poisson':
                # 泊松分布：方差=均值
                z = (col_data - mean) / std

            elif dist_type == 'binomial':
                # 二项分布：获取试验次数
                if isinstance(n_trials, dict):
                    n = n_trials.get(col, 10)  # 默认10次试验
                else:
                    n = n_trials or 10

                p = mean / n
                std = np.sqrt(n * p * (1 - p))
                z = (col_data - n * p) / std
            else:
                raise ValueError(f"不支持的离散分布类型: {dist_type}")

            col_cond = abs(z) > dyn_threshold
            outlier_cond = col_cond if outlier_cond is None else outlier_cond | col_cond

        self.manager.ddf = ddf[~outlier_cond]
        self.manager._persist_data()

    def _weibull_method(self, columns: List[str], threshold: float = 0.05):
        """Weibull分布异常检测"""
        from scipy.stats import weibull_min
        for col in columns:
            data = self.manager.ddf[col].compute()
            c, loc, scale = weibull_min.fit(data, floc=0)

            # 动态调节逻辑
            if c < 1:  # 早期失效模式
                dynamic_threshold = threshold * 1.5  # 放宽阈值
            elif c > 3:  # 严重磨损失效
                dynamic_threshold = threshold * 0.7  # 收紧阈值
            else:
                dynamic_threshold = threshold

            # 计算累积概率
            probs = weibull_min.cdf(data, c, loc, scale)
            mask = (probs > dynamic_threshold) & (probs < (1 - dynamic_threshold))
            self.manager.ddf = self.manager.ddf[mask]

    def _lognormal_method(self, columns: List[str], base_threshold: float = 0.05,
                          auto_adjust: bool = True, target_outlier_rate: float = 0.02):
        """对数正态分布异常检测"""
        from scipy.stats import lognorm
        for col in columns:
            data = self.manager.ddf[col].compute()

            s, loc, scale = lognorm.fit(data, floc=0)

            # 动态阈值计算
            if auto_adjust:
                dyn_threshold = self._calc_dynamic_threshold(
                    col_data=data,
                    base_threshold=base_threshold
                )
                # 异常率反馈控制
                current_threshold = self._feedback_adjustment(
                    data=data,
                    params=(s, loc, scale),  # 添加拟合参数元组
                    current_threshold=dyn_threshold,
                    target_rate=target_outlier_rate
                )
            else:
                current_threshold = base_threshold

            # 计算累积概率
            probs = lognorm.cdf(data, s, loc, scale)
            mask = (probs > current_threshold) & (probs < (1 - current_threshold))
            self.manager.ddf = self.manager.ddf[mask]

    # 优化处理
    def _optimized_quantile(self, columns: List[str],
                            quantiles=None,
                            force_chunk: bool = False):
        """
        分位数计算优化方案（内存/速度平衡）

        参数：
        force_chunk: 强制启用分块计算模式（默认False）
        """
        if quantiles is None:
            quantiles = [0.25, 0.75]

        # 智能方案选择
        if force_chunk or self.manager.ddf.npartitions > 100:

            from dask import delayed
            @delayed
            def chunk_quantile(chunk):
                return chunk.quantile(quantiles)

            results = []
            for chunk in self.manager.ddf[columns].to_delayed():
                results.append(chunk_quantile(chunk))

            computed = dd.compute(*results)
            q = pd.concat(computed).groupby(level=0).mean()
        else:

            q = self.manager.ddf[columns].quantile(
                quantiles,
                method='tdigest'
            ).compute()

        return q

    def _universal_dist_method(self, columns: List[str], dist_name: str, **params):
        """统一分布异常检测接口"""
        from functools import partial
        import inspect

        dist_handlers = {
            'normal': self._zscore_method,
            'poisson': partial(self._discrete_dist_method, dist_type='poisson'),
            'binomial': partial(self._discrete_dist_method, dist_type='binomial'),
            'lognormal': self._lognormal_method,
            'weibull': self._weibull_method
        }

        if dist_name not in dist_handlers:
            raise ValueError(f"不支持的分布类型: {dist_name}")

        handler = dist_handlers[dist_name]
        sig = inspect.signature(handler)
        valid_params = {k: v for k, v in params.items() if k in sig.parameters}

        try:
            handler(columns, **valid_params)
        except TypeError as e:
            raise ValueError(f"参数不匹配: {str(e)}")

    def _auto_eps_kdist(self, columns: List[str], k: int = 10) -> float:
        """通过K-距离图自动选择eps（分布式计算优化）"""
        import dask.array as da
        import numpy as np

        data = self.manager.ddf[columns].to_dask_array(lengths=True)  # 转为Dask Array
        sample_size = min(10000, len(data))  # 动态采样大小

        if len(data) > sample_size:
            random_indices = da.random.choice(len(data), size=sample_size, replace=False)
            data = data[random_indices]

        pairwise_dist = da.linalg.norm(data[:, None] - data, axis=2)
        k_distances = da.apply_along_axis(
            lambda x: np.partition(x, k)[k],
            axis=1,
            arr=pairwise_dist
        )

        # 聚合并排序
        sorted_dists = np.sort(k_distances.compute())

        # 自动检测拐点（优化阈值检测）
        diffs = np.diff(sorted_dists)
        eps_threshold = np.percentile(diffs, 85)  # 调优至85%分位点
        eps_index = np.argmax(diffs < eps_threshold)
        eps = float(sorted_dists[eps_index])

        return eps

    def _auto_eps_sampling(self, columns: List[str], sample_size: int = 10000, k: int = 10) -> float:
        """通过随机采样估计eps（适用于超大数据集）
        基于随机采样的轻量级适配"""
        sample = self.manager.ddf[columns].sample(n=sample_size).compute()

        # 计算采样点的k近邻距离
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(sample)
        dists = nn.kneighbors()[0]
        k_distances = dists[:, -1]

        # 排序并检测拐点
        sorted_dists = np.sort(k_distances)
        diffs = np.diff(sorted_dists)
        eps = sorted_dists[np.argmax(diffs < np.mean(diffs))]  # 取低于平均差分的首个点

        return float(eps)

    def _calc_dynamic_threshold(self, col_data, base_threshold: float = 3):
        """阈值的自动调整"""

        class DynamicThresholdOptimizer:
            def __init__(self, base_threshold: float = 3):
                self.base_threshold = base_threshold

            def _lightweight_monitor(self, s, data_size):
                """阶段1：轻量级实时监控"""
                s_adjust = 1 + max(0, (s - 1) / 2)
                size_factor = np.log10(max(100, data_size)) / 2
                return self.base_threshold * s_adjust / size_factor

            def _robust_calculation(self, col_data):
                """阶段2：周期性深度计算"""
                # 使用近似计算加速（分位数采样）
                sample_data = col_data.sample(frac=0.2, random_state=42) if len(col_data) > 1e4 else col_data
                q75, q25 = sample_data.quantile([0.75, 0.25]).compute()
                iqr = q75 - q25
                mad = (sample_data - sample_data.median()).abs().median().compute() * 1.4826
                return 0.7 * mad + 0.3 * iqr  # 返回volatility_scale

            def get_threshold(self, col_data, s=None, force_full=False):
                """动态阈值生成入口"""
                data_size = len(col_data)

                # 阶段1：快速生成基线阈值
                base_thresh = self._lightweight_monitor(s, data_size) if s else self.base_threshold

                # 阶段2：按需触发深度计算
                if force_full or data_size > 5000:  # 大数据集或强制要求时
                    volatility_scale = self._robust_calculation(col_data)
                    skewness = col_data.skew().compute()
                    adjust_factor = 1 + min(1.0, abs(skewness) / 3)  # （偏度>3时阈值扩大33%）
                    fdr_factor = np.log(max(10, data_size))
                    return base_thresh * volatility_scale * adjust_factor * fdr_factor
                return base_thresh

        compute_threshold = DynamicThresholdOptimizer(base_threshold)
        return compute_threshold.get_threshold(col_data)

    def _calc_dynamic_range(self, col_data, base_range, historical_rate, target_rate):
        """动态range_综合计算"""
        # 策略1：偏度调整
        range_by_skew = self._adjust_range_based_on_skewness(col_data)

        # 策略2：实时异常率反馈
        range_by_feedback = self._feedback_adjust_range(
            current_rate=historical_rate,
            target_rate=target_rate,
            base_range=base_range
        )

        # 融合策略：加权平均
        return 0.7 * range_by_skew + 0.3 * range_by_feedback

    def _adjust_range_based_on_skewness(self, col_data):
        """基于偏度的范围调整"""
        try:
            skewness = col_data.skew().compute()
            abs_skew = abs(skewness)

            # 偏度调整逻辑：偏度越大，范围越宽松
            if abs_skew < 0.5:  # 接近对称分布
                return 1.3
            elif abs_skew < 1.0:  # 轻度偏态
                return 1.5
            elif abs_skew < 2.0:  # 中度偏态
                return 1.8
            else:  # 重度偏态
                return min(3.0, 1.5 + (abs_skew - 1.5) * 0.5)
        except:
            return 1.5  # 计算失败时返回默认值

    def _feedback_adjust_range(self, current_rate, target_rate, base_range):
        """PID式异常率反馈控制"""
        # 计算异常率偏差
        rate_error = current_rate - target_rate

        # PID控制参数（比例、积分、微分系数）
        Kp = 0.2  # 比例系数
        Ki = 0.05  # 积分系数
        Kd = 0.1  # 微分系数

        # 初始化历史误差记录
        if not hasattr(self, '_rate_errors'):
            self._rate_errors = []

        # 保存当前误差（最多保留5个历史值）
        self._rate_errors.append(rate_error)
        if len(self._rate_errors) > 5:
            self._rate_errors.pop(0)

        # 计算PID分量
        P = Kp * rate_error
        I = Ki * sum(self._rate_errors)
        D = Kd * (rate_error - self._rate_errors[-2]) if len(self._rate_errors) > 1 else 0

        # 计算调整量并限制范围
        adjustment = P + I + D
        new_range = base_range + adjustment

        # 限制范围在合理区间[0.5, 3.0]
        return max(0.5, min(3.0, new_range))

    def _calc_dynamic_z_threshold(self, col_data, base_threshold: float = 3):
        """Z-score专用动态阈值优化"""
        # 鲁棒波动性度量（抗离群值）
        mad = (col_data - col_data.median()).abs().median().compute() * 1.4826
        q1, q3 = col_data.quantile([0.25, 0.75]).compute()
        iqr = q3 - q1
        volatility = 0.7 * mad + 0.3 * iqr

        # 偏度修正（右偏数据需放宽阈值）
        skew = col_data.skew().compute()
        skew_factor = 1 + min(1.0, abs(skew) / 4)  # 偏度>4时阈值扩大25%

        # 数据量修正（小样本收紧阈值）
        size_factor = np.log10(max(100, len(col_data))) / 2

        return base_threshold * volatility * skew_factor / size_factor

    def _score_based_contamination(self, model, columns: List[str], safety_margin=0.05):
        """优化版阈值动态计算"""
        scores = model.decision_function(self.manager.ddf[columns])
        if hasattr(scores, 'compute'):  # Dask数组处理
            scores = scores.compute()


        scores = np.asarray(scores).ravel()

        # 1. 直方图突变检测
        hist, edges = np.histogram(scores, bins=min(100, len(scores) // 10))
        diff = np.diff(hist)
        if np.all(diff == 0):
            threshold1 = edges[0]
        else:
            drop_idx = np.argmax(diff < -np.std(diff))
            threshold1 = edges[drop_idx]

        # 2. 偏度自适应调整
        from scipy.stats import skew
        skew_result = np.asarray(skew(scores))
        if skew_result.ndim == 0:
            skewness_val = float(skew_result)
        elif skew_result.size == 1:
            skewness_val = float(skew_result.item())
        else:  # 多维数组
            skewness_val = float(skew_result.flat[0])
        skew_factor = 1 + min(1.0, abs(skewness_val) / 4)  # 偏度>4时扩大25%
        threshold2 = np.percentile(scores, 5) * skew_factor  # 底部5%分位

        # 3. 波动率加权
        scores_series = pd.Series(scores)
        rolling_std = scores_series.rolling(500).std().dropna()
        volatility = max(0, rolling_std.mean()) if rolling_std.size > 0 else max(0, np.std(scores))
        threshold3 = np.mean(scores) - 2.5 * volatility

        # 融合阈值
        final_threshold = min(
            float(threshold1),
            float(threshold2),
            float(np.real(threshold3))
        )

        # 计算异常比例
        prop = (scores < final_threshold).mean()
        return max(0.01, prop + safety_margin)

    @staticmethod
    def _feedback_adjustment(data, params, current_threshold, target_rate):
        """PID式异常率反馈"""
        from scipy.stats import lognorm
        probs = lognorm.cdf(data, *params)
        outlier_rate = ((probs < current_threshold) | (probs > 1 - current_threshold)).mean()
        adjust_step = 0.3 * (outlier_rate - target_rate)
        return np.clip(current_threshold + adjust_step, 0.01, 0.25)

    # 分支处理
    def _generate_outlier_mask(self):
        """生成与原始数据对齐的异常值掩码"""
        if self._original_ddf is None or self._cleaned_ddf is None:
            return

        original_computed = self._original_ddf.compute()
        cleaned_computed = self._cleaned_ddf.compute()

        aligned_original, aligned_cleaned = original_computed.align(
            cleaned_computed,
            join='left',
            fill_value=None
        )

        mask = aligned_cleaned.isna()
        self._outlier_mask = dd.from_pandas(
            mask,
            npartitions=self._original_ddf.npartitions
        )

    # 数据访问接口
    def get_original_data(self) -> dd.DataFrame:
        """获取原始数据（独立副本）"""
        return self._original_ddf.copy() if self._original_ddf is not None else None

    def get_clean_data(self) -> dd.DataFrame:
        """获取清洗后的数据（独立副本）"""
        return self._cleaned_ddf.copy() if self._cleaned_ddf is not None else self.manager.ddf

    def get_outlier_mask(self) -> dd.DataFrame:
        """获取异常值标记矩阵"""
        return self._outlier_mask.copy() if self._outlier_mask is not None else None


from Visualization.rectangle import HeatmapGenerator, DataVisualizer
import matplotlib.patches as mpatches


class DrawData:
    """大数据可视化引擎"""

    def __init__(self, manager: DataManager):
        self.manager = manager
        self.heatmap_generator = None  # 热力图生成器实例
        self.visualizer = DataVisualizer(style="whitegrid", palette="pastel", figsize=(12, 8))

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
        elif plot_type == 'heatmap':
            return self.draw_heatmap_with_outliers(**kwargs)
        else:
            raise ValueError(f"不支持的图表类型: {plot_type}")

    def _plot_histogram(self, ddf: dd.DataFrame, column: str, bins: int = 100):
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

    def _plot_boxplot(self, ddf: dd.DataFrame, column: str):
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

    def _plot_qq(self, ddf: dd.DataFrame, column: str, sample_size: int = 5000):
        """Q-Q图（优化抽样方法）
        用概率图（如Q-Q图）来辅助判断数据分布类型"""
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
        集成异常值染色的热力图生成
        :param outlier_handler: 异常值处理器实例
        :param highlight_outliers: 是否高亮显示异常值
        :param kwargs: 透传给create_heatmap的参数
        """
        # 生成基础热力图
        hm = self._create_heatmap(outlier_handler, **kwargs)

        # 自动判断是否需要异常值染色
        should_highlight = (
                highlight_outliers and
                outlier_handler is not None and
                outlier_handler.get_outlier_mask() is not None
        )

        if should_highlight:
            # 获取异常值掩膜
            outlier_mask = outlier_handler.get_outlier_mask().compute()

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
                        handling_outlier: HandlingOutlier,
                        data_type: Literal['raw', 'cleaned'] = 'raw',  # 原始数据（raw）或清洗后数据（cleaned）
                        columns: List[str] = None,  # columns: 指定分析的列名（默认全量）
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
        # 获取数据源
        data_source = self.manager.ddf if data_type == 'raw' \
            else handling_outlier.get_clean_data()

        # 计算相关性矩阵
        corr_data = data_source[columns].corr().compute() if columns \
            else data_source.corr().compute()

        self.heatmap_generator = HeatmapGenerator(
            corr_data,
            title=title,
            color_map=color_map,
            annot=True
        )
        return self.heatmap_generator

    def _get_outlier_mask(self, handler: HandlingOutlier) -> pd.DataFrame:
        """生成索引对齐的布尔掩膜矩阵"""
        # 获取原始数据和清洗后数据
        original = self.manager.ddf.compute()
        cleaned = handler.get_clean_data().compute()

        aligned_original, aligned_cleaned = original.align(cleaned, join='outer', fill_value=np.nan)

        # 生成布尔矩阵（使用不同的变量名避免混淆
        outlier_mask = (aligned_original.fillna(0) != aligned_cleaned.fillna(0))  # 通过填充缺失值为 0 后比较生成布尔掩膜

        # 正确使用 fillna 处理缺失值（在 DataFrame 上操作）
        return outlier_mask.fillna(False)

    # 新增：异常值处理前的可视化
    def visualize_pre_processing(self, handler: HandlingOutlier, columns: List[str]):
        """异常值处理前数据可视化"""
        if not columns:
            return

        print("\n" + "=" * 50)
        print("异常值处理前数据分布可视化")
        print("=" * 50)

        ddf = handler.get_original_data() or self.manager.ddf

        for col in columns:
            try:
                sample = ddf[col].sample(frac=0.1).compute() if ddf.npartitions > 1 else ddf[col].compute()
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # 直方图
                self.visualizer.hist_plot(
                    data=pd.DataFrame({col: sample}),
                    column=col,
                    title=f'【{col}】处理前分布',
                    xlabel='值',
                    ylabel='频数',
                    ax=axes[1]
                )
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"列 {col} 可视化失败: {str(e)}")

    # 新增：异常值处理后的可视化
    def visualize_post_processing(self, handler: HandlingOutlier, columns: List[str]):
        """异常值处理后数据可视化"""
        if not columns:
            return

        print("\n" + "=" * 50)
        print("异常值处理后数据分布可视化")
        print("=" * 50)

        ddf = handler.get_clean_data()

        for col in columns:
            try:
                sample = ddf[col].sample(frac=0.1).compute() if ddf.npartitions > 1 else ddf[col].compute()
                fig, axes = plt.subplots(1, 2, figsize=(15, 6))

                # 直方图
                self.visualizer.hist_plot(
                    data=pd.DataFrame({col: sample}),
                    column=col,
                    title=f'【{col}】处理后分布',
                    xlabel='值',
                    ylabel='频数',
                    ax=axes[1]
                )
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"列 {col} 可视化失败: {str(e)}")

    # 新增：异常值标记可视化
    @staticmethod
    def visualize_outliers(handler: HandlingOutlier, columns: List[str] = None):
        """异常值分布可视化"""
        if not handler.get_original_data() or not handler.get_outlier_mask():
            print("无法可视化异常值：缺少原始数据或异常值掩码")
            return

        original = handler.get_original_data()
        mask = handler.get_outlier_mask()

        if columns is None:
            columns = mask.columns.tolist()

        print("\n" + "=" * 50)
        print("异常值分布可视化")
        print("=" * 50)

        for col in columns:
            try:
                original_data = original[col].compute()
                outlier_mask = mask[col].compute()

                plt.figure(figsize=(12, 6))
                plt.scatter(
                    x=range(len(original_data)),
                    y=original_data,
                    c=outlier_mask.map({True: 'red', False: 'blue'}),
                    alpha=0.6,
                    s=20
                )
                plt.title(f'【{col}】异常值分布（红色为异常值）')
                plt.xlabel('数据索引')
                plt.ylabel('值')
                plt.legend(handles=[
                    mpatches.Patch(color='blue', label='正常值'),
                    mpatches.Patch(color='red', label='异常值')
                ])
                plt.grid(alpha=0.3)
                plt.show()
            except Exception as e:
                print(f"列 {col} 异常值可视化失败: {str(e)}")

class JudgeMethod:
    """分布类型判断引擎（优化大样本处理）"""

    def __init__(self, manager: DataManager):
        self.manager = manager

    def judge_distribution(self, column: str) -> dict:
        """主入口方法：判断列数据的分布类型"""
        if column not in self.manager.ddf.columns:
            raise ValueError(f"列 {column} 不存在于数据集中")

        # 检查数据类型
        dtype = self.manager.ddf[column].dtype
        if np.issubdtype(dtype, np.number):

            # 数值型数据：检查是连续还是离散
            nunique = self.manager.ddf[column].nunique().compute()
            total = self.manager.ddf[column].count().compute()

            discrete_threshold = max(20, total * 0.05)  # 动态阈值：至少20个唯一值或5%比例
            is_integer = np.issubdtype(dtype, np.integer)  # 检查是否为整数类型

            # 判断连续型或离散型
            if nunique >= discrete_threshold and not (is_integer and nunique <= 50):
                return self._judge_continuous(column)
            else:
                discrete_series = self.manager.ddf[column].compute()
                return self._judge_discrete(discrete_series)  # 离散型分布
        else:
            # 非数值型数据直接返回分类
            return {'distribution': 'categorical'}

    def _judge_continuous(self, column: str) -> dict:
        """综合改进的连续型分布判断"""
        try:
            # 使用近似计算加速
            total = self.manager.ddf[column].count().compute()

            # 并行计算偏度和峰度
            skew_kurt = self.manager.ddf[column].agg(['skew', 'kurtosis'],
                                                     split_every=10).compute()
            skewness = skew_kurt['skew']  # 偏度值
            kurtosis = skew_kurt['kurtosis']  # 峰度值

            # 改进的分布类型判断
            distribution_type = self._determine_dist_type(skewness, kurtosis)

            # 分层抽样策略（优化尾部数据捕捉）
            sample_size = self._get_sample_size(total)
            sample = self._stratified_sampling(column, sample_size)

            # 多方法正态检验
            norm_info = self._run_normality_tests(sample)
            is_normal = self._judge_normality(norm_info, len(sample))

            # 非正态数据的深入分析
            if not is_normal and distribution_type == 'non-normal_complex':
                # 步骤1：分布拟合
                fitted_dist = self._fit_distributions(
                    sample)
                if fitted_dist != 'unknown':
                    distribution_type = fitted_dist

                # 步骤2：特殊形状检测
                shape_type = self._detect_j_u_shapes(sample)
                if shape_type:
                    distribution_type = shape_type

            return {
                'distribution': 'normal' if is_normal else distribution_type,
                'skew_type': self._classify_skew(skewness),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis),
                'norm_tests': norm_info,
                'sample_size': sample_size
            }
        except Exception as e:
            print(f"分布判断失败: {str(e)}")
            return {'error': str(e)}

    def _stratified_sampling(self, column: str, sample_size: int) -> pd.Series:
        """分层抽样（保证尾部数据）"""
        if sample_size <= 0:
            raise ValueError("抽样大小必须为正整数")

        ddf = self.manager.ddf
        # 使用1%/99%分位数防止极端值影响
        quantiles = ddf[column].quantile([0.01, 0.25, 0.5, 0.75, 0.99]).compute()

        # 添加最小值/最大值确保全区间覆盖
        min_val = ddf[column].min().compute()
        max_val = ddf[column].max().compute()
        bins = [min_val - 1e-6]
        bins.extend(quantiles)
        bins.append(max_val + 1e-6)

        # 计算各分位数区间的数量
        labels = [f'Q{i}' for i in range(len(bins) - 1)]
        ddf = ddf.assign(strata=lambda df: pd.cut(df[column], bins=bins, labels=labels))
        strata_counts = ddf['strata'].value_counts().compute()
        total = strata_counts.sum()

        # 分配抽样数量
        samples = []
        for stratum, count in strata_counts.items():
            proportion = count / total
            n_samples = max(1, int(sample_size * proportion))
            if n_samples > count:
                n_samples = count

            sampled = ddf[ddf['strata'] == stratum].sample(n=n_samples, random_state=42)
            samples.append(sampled)

        combined = dd.concat(samples).compute()[column]
        return combined

    def _streaming_statistics(self, column: str):
        """TB级数据流式统计（优化增量计算）"""
        if not hasattr(self, '_stream_cache'):
            self._stream_cache = {}

        cache = self._stream_cache.get(column, {'count': 0, 'mean': 0.0, 'M2': 0.0})
        self._stream_cache[column] = cache  # 确保引用更新

        for part in self.manager.ddf[column].partitions:
            part_data = part.compute()
            for x in part_data:
                # 修正Welford算法(二次项计算)
                count = cache['count']
                mean = cache['mean']

                count += 1
                delta = x - mean
                new_mean = mean + delta / count

                # 修复公式: delta * (x - new_mean) 替代原错误公式
                cache['M2'] += delta * (x - new_mean)
                cache['mean'] = new_mean
                cache['count'] = count

        # 计算最终统计量
        count = cache['count']
        mean = cache['mean']
        variance = cache['M2'] / (count - 1) if count > 1 else 0
        std_dev = np.sqrt(variance)
        print(f"实时统计量[增量] - {column}: μ={mean:.2f}, σ={std_dev:.2f}, n={count:,}")

    @staticmethod
    def _fit_distributions(sample: pd.Series) -> str:
        """拟合多种分布并选择最佳匹配（基于KS检验和AIC）"""
        distributions = [
            ('expon', stats.expon),  # 指数分布
            ('gamma', stats.gamma),  # Gamma分布
            ('lognorm', stats.lognorm),  # 对数正态分布
            ('weibull_min', stats.weibull_min),  # Weibull分布
            ('uniform', stats.uniform),  # 均匀分布
            ('chi2', stats.chi2),  # 卡方分布
        ]

        best_fit = ""
        min_aic = float('inf')

        for name, dist in distributions:
            try:
                # 卡方分布特殊处理：自由度应为正整数
                if name == 'chi2':
                    if sample.min() < 0:
                        continue

                    df, loc, scale = dist.fit(sample, floc=0)
                    if df <= 0:  # 无效自由度
                        continue
                    params = (df, loc, scale)
                    cdf = dist.cdf
                else:
                    params = dist.fit(sample)
                    cdf = dist.cdf

                # 计算PDF和AIC
                pdf = dist.pdf(sample, *params)
                log_likelihood = np.sum(np.log(pdf + 1e-9))
                aic = 2 * len(params) - 2 * log_likelihood

                # KS检验
                _, ks_p = stats.kstest(sample, cdf, args=params)
                # 增加AD检验作为补充
                theoretical_sample = dist.rvs(size=1000, *params)
                _, ad_p = stats.anderson_ksamp([sample, theoretical_sample])

                # 综合评估标准
                fit_quality = 0
                if ks_p > 0.05: fit_quality += 1
                if ad_p > 0.05: fit_quality += 1
                if aic < min_aic * 1.1: fit_quality += 1  # 允许10%浮动

                # 选择最佳拟合
                if fit_quality >= 2 and aic < min_aic:
                    min_aic = aic
                    best_fit = name

            except Exception as e:
                print(f"拟合时报错{e}")
                continue

        return best_fit or 'unknown'

    @staticmethod
    def _detect_j_u_shapes(sample: pd.Series) -> str:
        """检测J型/U型分布（结合分位数特征和核密度峰值检测）"""
        # 初始化变量
        x = np.array([])
        peaks = []

        # 计算分位数比例
        q25, q50, q75 = np.percentile(sample, [25, 50, 75])
        left_ratio = len(sample[sample < q25]) / len(sample)
        right_ratio = len(sample[sample > q75]) / len(sample)
        mid_ratio = 1 - (left_ratio + right_ratio)

        # 核密度估计峰值检测
        try:
            from scipy.stats import gaussian_kde
            sample_array = sample.dropna().values
            if len(sample_array) >= 10:
                percentiles = np.percentile(sample_array, [75, 25])
                iqr = float(percentiles[1] - percentiles[0])
                std_dev = float(np.std(sample_array))

                bw_factor = min(std_dev, iqr / 1.34)
                n = len(sample_array)
                bw = 0.9 * bw_factor * n ** (-0.2)

                kde = gaussian_kde(sample_array, bw_method=bw)
                min_val, max_val = np.min(sample_array), np.max(sample_array)
                x = np.linspace(min_val, max_val, 500)

                density = kde(x)
                grad1 = np.gradient(density, x)
                grad2 = np.gradient(grad1, x)
                peak_candidates = np.where((grad2 < 0) & (np.diff(np.sign(grad1), prepend=0) < 0))[0]

                if len(peak_candidates) > 0:
                    main_peak_height = np.max(density[peak_candidates])
                    peaks = [p for p in peak_candidates if density[p] > 0.2 * main_peak_height]
        except Exception:
            pass

        num_peaks = len(peaks)

        # U型分布判断
        if left_ratio > 0.3 and right_ratio > 0.3 and mid_ratio < 0.4:
            if num_peaks == 2 and len(x) > 0:
                left_peak = any(x[p] < q25 for p in peaks)
                right_peak = any(x[p] > q75 for p in peaks)
                if left_peak and right_peak:
                    return 'u_shape'
            return ""

        # J型分布判断
        elif left_ratio < 0.2 and right_ratio > 0.5:
            return 'j_shape'
        elif left_ratio > 0.5 and right_ratio < 0.2:
            return 'reverse_j_shape'

        return ""

    @staticmethod
    def _get_sample_size(total: int) -> int:
        """动态样本量策略"""
        if total > 1e8:
            return 100_000
        elif total > 1e6:
            return 10_000
        return min(5000, total)

    @staticmethod
    def _run_normality_tests(sample: pd.Series) -> dict:
        """综合正态性检验"""
        norm_info = {}

        if len(sample) < 50:
            # 小样本优先使用Shapiro-Wilk
            _, p_shapiro = stats.shapiro(sample)
            norm_info['shapiro_p'] = p_shapiro
        elif 50 <= len(sample) <= 5000:
            # 中等样本使用D'Agostino+Shapiro
            _, p_normal = normaltest(sample)
            _, p_shapiro = stats.shapiro(sample)
            norm_info.update({'dagostino_p': p_normal, 'shapiro_p': p_shapiro})
        else:
            # 大样本使用D'Agostino+Anderson
            _, p_normal = normaltest(sample)
            anderson_result = anderson(sample, dist='norm')
            norm_info.update({
                'dagostino_p': p_normal,
                'anderson_stat': anderson_result.statistic,
                'critical_values': anderson_result.critical_values
            })
        return norm_info

    @staticmethod
    def _judge_normality(norm_info: dict, sample_size: int) -> bool:
        """综合判断正态性（增强鲁棒性）"""
        # 根据样本量选择检验标准
        if sample_size < 50:
            return norm_info.get('shapiro_p', 0) > 0.05

        elif 50 <= sample_size <= 5000:
            shapiro_pass = norm_info.get('shapiro_p', 0) > 0.05
            dagostino_pass = norm_info.get('dagostino_p', 0) > 0.05
            return shapiro_pass and dagostino_pass

        else:
            # Anderson检验：统计量小于5%临界值
            anderson_stat = norm_info.get('anderson_stat', float('inf'))
            crit_val = norm_info.get('critical_values', [])[2] if len(
                norm_info.get('critical_values', [])) > 2 else float('inf')
            dagostino_pass = norm_info.get('dagostino_p', 0) > 0.05

            return (anderson_stat < crit_val) and dagostino_pass

    @staticmethod
    def _determine_dist_type(skewness: float, kurtosis: float) -> str:
        """基于矩量的分布类型判断（增强版）"""
        abs_skew = abs(skewness)
        direction = "right" if skewness > 0 else "left"
        tail_type = "heavy" if kurtosis > 3 else "light"

        # 峰度偏离度（相对于正态分布）
        kurtosis_dev = abs(kurtosis - 3)

        if abs_skew < 0.5 and kurtosis_dev < 1:
            return 'normal-like'
        elif abs_skew > 1 and kurtosis_dev > 2:
            return f'{direction}_skew_{tail_type}_tail'
        elif (0.5 <= abs_skew <= 1) and (kurtosis_dev > 1):
            return f'{direction}_skew_moderate'
        elif kurtosis_dev > 3:
            return f'leptokurtic_{direction}'  # 尖峰分布
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
        nunique = series.nunique()
        result = {'distribution': 'categorical' if nunique < 10 else 'multimodal'}

        # 二项分布检测
        if series.min() >= 0 and series.max() <= 1 and nunique == 2:
            p_est = series.mean()
            result.update({'distribution': 'binomial', 'p': p_est})
            return result

        # 泊松分布检测
        if series.min() >= 0 and pd.api.types.is_integer_dtype(series):
            lambda_est = series.mean()
            value_counts = series.value_counts().sort_index()
            obs_values = value_counts.index.values
            f_obs = value_counts.values

            f_exp = stats.poisson.pmf(obs_values, lambda_est) * len(series)

            # 合并小期望频数分组（卡方检验前提）
            while len(f_exp) > 3 and np.any(f_exp < 5):
                min_idx = np.argmin(f_exp)
                merge_idx = min_idx - 1 if min_idx == len(f_exp) - 1 else min_idx + 1

                # 合并相邻分组
                f_obs[min_idx] += f_obs[merge_idx]
                f_exp[min_idx] += f_exp[merge_idx]

                # 删除被合并项
                f_obs = np.delete(f_obs, merge_idx)
                f_exp = np.delete(f_exp, merge_idx)
                obs_values = np.delete(obs_values, merge_idx)

            # 执行检验
            if len(f_exp) >= 3 and np.all(f_exp >= 1) and (f_exp >= 5).sum() / len(f_exp) > 0.8:
                _, p_value = stats.chisquare(f_obs, f_exp)
            else:
                _, p_value = stats.fisher_exact(np.vstack([f_obs, f_exp]).T)

            if p_value > 0.05:
                result['distribution'] = 'poisson'
                result['lambda'] = lambda_est

                # 负二项分布检测（过度离散）
                variance = series.var()
                if variance > 1.5 * lambda_est:
                    result['distribution'] = 'negative_binomial'
                    result['dispersion'] = variance / lambda_est

        return result


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
