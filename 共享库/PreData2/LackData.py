# 缺失值
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from pathlib import Path
from typing import TypedDict, List, Union,Optional,Callable
import numpy as np
from dask_ml.impute import SimpleImputer
from dask_ml.wrappers import ParallelPostFit  # 并行包装器
from sklearn.experimental import enable_iterative_imputer  # 新增
from sklearn.impute import IterativeImputer  # 修正导入
from sklearn.ensemble import RandomForestRegressor  # 新增导入
import gc
import psutil  # 新增内存监控依赖
# import dask  # 新增Dask核心库依赖
from dask.distributed import wait
import warnings

# 监控Dask仪表板（默认端口8787）

class Settings(TypedDict):
    path: Path
    blocksize: str
    na_value: Union[str, List[str]]
    dtype: dict
    compression: str  # 压缩方式
    output_format: str  # 补充缺失的类型声明

class ManagerSettings(Settings,total=False):    # 父类字段仍为必填
    time_column: str        # 新增时间列配置           time_column: NotRequired[str]        # 选填字段
    time_interval: Union[str,None]   # 时间间隔  等价于 time: str | None  和   time: Optional[str]


class DataManager:
    def __init__(self, setting: ManagerSettings,save:bool=False,n_workers: int=4,memory_limit:str = '1GB',radical:bool =False):
        """
        :param setting:
        :param save:
        :param n_workers:
        :param memory_limit:
        :param radical:
        注册管理器
        """
        self.setting = setting
        self.save = save
        self.strategy = None

        # 初始化Dask客户端（可选）
        self.client = Client(n_workers=n_workers, memory_limit=memory_limit)
        # 指定使用4个Worker进程，每个Worker的内存上限为1GB。这相当于将本地机器的计算资源划分为4个并行单元
        # 通过Client对象自动协调任务的分配和执行，实现多核并行计算
        # 监控支持：启动后会生成Web监控界面（通过client.dashboard_link查看），可实时观察任务进度和资源利用率
        self.ddf = None  # 新增：存储加载的Dask DataFrame

        # 新增垃圾回收计数器
        self.gc_counter = 0
        self.max_gc_counter = 3

        if radical:
            self.optimize_dask_config()  # 添加此行

    def _infer_dtypes(self):
        """使用Dask内置的元数据推断"""
        try:
            # 尝试读取首块数据推断类型
            ddf_sample = dd.read_csv(
                self.setting['path'],
                blocksize="10MB"
            )
            return dd.utils.make_meta(ddf_sample).dtypes
        except Exception as e:
            # 回退到原有逻辑
            print(f"自动推断失败: {str(e)}, 使用备用方法")
            sample = dd.read_csv(
                self.setting['path'],
                blocksize="10MB"
            ).head(n=1000)
            return {
                col: 'category' if sample[col].nunique() < min(1000, len(sample) // 10)
                else pd.api.types.infer_dtype(sample[col])
                for col in sample.columns
            }

    def load_data(self):
        """支持多格式数据加载"""
        try:
            path = str(self.setting['path'])
            dtype = self.setting.get('dtype', self._infer_dtypes())

            if path.endswith('.parquet'):
                self.ddf = dd.read_parquet(
                    path,
                    engine='pyarrow',
                    dtype=dtype
                )
            else:
                self.ddf = dd.read_csv(
                    path,
                    blocksize=self.setting['blocksize'],
                    na_values=self.setting['na_value'],
                    dtype=dtype,
                    assume_missing=True,
                    on_bad_lines='warn'
                )

        except Exception as e:
            print(f"数据加载失败，报错: {str(e)}")
            # raise RuntimeError("数据加载失败")  # 合法写法
            raise  # 重新抛出当前捕获的异常



    def smart_gc(self, force=False) -> bool:
        """智能垃圾回收策略（支持TB级数据）"""
        workers_mem = self.client.run(lambda: psutil.virtual_memory().percent)

        if force or any(p > 75 for p in workers_mem.values()):
            print(f"触发紧急内存回收（Worker内存使用率：{workers_mem}）")

            # 取消所有未完成的Future对象（正确类型）
            self.client.cancel(self.client.futures)  # 直接获取活跃的Future列表

            # 触发分布式垃圾回收
            self.client.run(lambda: gc.collect(generation=2))

            # 极端内存压力下重启Worker
            if any(p > 90 for p in workers_mem.values()):
                print("执行Worker滚动重启并重载数据")
                self.client.restart()
                self.ddf = None  # 清除旧引用
                self.load_data()  # 重启后重新加载数据

            return True
        return False

    def select_strategy(self, strategy_selector: Optional[Callable[[str,list[str]],Optional[str]]] = None):
        """为每个字段选择具体的处理策略

        :param strategy_selector: 自定义策略选择函数，格式为 func(col: str, strategies: List[str]) -> str
                                   若不提供则默认选择每个字段的第一个可用策略
        """


        if not self.strategy:
            raise ValueError("请先通过HandleStrategy生成策略选项")

        # 设置默认选择器（选择第一个策略）
        if strategy_selector is None:
            strategy_selector = lambda col, strategies: strategies[0] if strategies else None

        # 创建策略副本避免修改原始数据
        selected_strategy = {}

        for col, strategies in self.strategy.items():
            if not isinstance(strategies, list):
                raise TypeError(f"列{col}的策略类型错误，应为列表，实际为{type(strategies)}")

            if not strategies:
                warnings.warn(f"列 {col} 无可用策略，将跳过处理")
                continue

            selected = strategy_selector(col, strategies)

            if selected not in strategies:
                raise ValueError(f"为列 {col} 选择策略 '{selected}' 不在可用策略列表 {strategies} 中")

            selected_strategy[col] = selected

        # 原子性更新策略
        self.strategy = selected_strategy

        # 清理无效字段
        self.strategy = {k: v for k, v in self.strategy.items() if v is not None}

    @staticmethod
    def optimize_dask_config():
        """Dask性能优化核心配置"""
        from dask.config import set
        set({
            "optimization.fuse.active": True,  # 启用任务融合
            "distributed.worker.memory.target": 0.7,  # 更早触发数据溢出
            "distributed.worker.memory.spill": 0.8,  # 更激进的内存管理
            "dataframe.shuffle.compression": "zstd",  # 使用高效压缩算法
            "admin.tick.limit": "5s"  # 控制调度开销
        })


class MissingValueAnalyzer:
    def __init__(self, manager: DataManager):
        self.manager = manager
        if not self.manager.ddf:
            self.manager.load_data()

    def _validate_parameters(self):
        """参数验证与类型转换"""
        # 确保na_value始终为列表类型
        if isinstance(self.manager.setting.get('na_value', []), str):
            self.manager.setting['na_value'] = [self.manager.setting['na_value']]

        # 自动推断blocksize（如果未指定）
        if not self.manager.setting.get('blocksize'):
            self.manager.setting['blocksize'] = '128MB'

    def plot_missing_rates(self, result:pd.DataFrame):
        """绘制缺失率水平柱状图（面向大型数据集优化版）"""
        import matplotlib.pyplot as plt
        # 配置可视化参数
        plt.style.use('seaborn')  # 使用更美观的主题[3](@ref)
        fig, ax = plt.subplots(figsize=(12, max(8, int(len(result) * 0.4)+1)))  # 动态调整高度

        # 创建水平柱状图（y轴显示列名更清晰）[5](@ref)
        bars = ax.barh(
            y=result.index.tolist(),
            width=result['Missing_Ratio(%)'] / 100,  # 转换为0-1比例
            color='#2c7fb8',  # 专业图表配色
            alpha=0.7
        )

        # 添加数据标签（显示百分比）[1](@ref)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{width:.1%}',  # 百分比格式
                    va='center')

        # 坐标轴配置
        ax.set_xlim(0, 1.1)  # 留出标签空间[8](@ref)
        ax.set_xlabel('Missing Value Ratio', fontsize=12)
        ax.set_title('Missing Values Distribution by Column', fontsize=14)
        ax.grid(axis='x', linestyle='--', alpha=0.7)  # 辅助网格线

        # 保存与显示
        plt.tight_layout()
        if self.manager.save:
            plt.savefig('missing_values_plot.png', dpi=300, bbox_inches='tight')
        plt.show()

    def analyze_missing_values(self) -> pd.DataFrame:
        """核心分析逻辑"""
        self._validate_parameters()

        # 使用已加载的数据（替代原有的重复加载逻辑）
        ddf = self.manager.ddf

        '''
        当CSV文件中存在字段数量不一致的异常行时（如某行多出/缺少列），'warn'会跳过这些行并输出警告，而非直接报错中断程序
        替代选项：其他可选值包括'error'（严格报错）或'skip'（静默跳过）
        '''

        # 计算缺失统计
        missing_counts = ddf.isnull().sum().compute()
        total_rows = ddf.shape[0].compute()
        missing_ratio = (missing_counts / total_rows * 100).round(2)

        '''
        ddf.isnull().sum()：对Dask DataFrame每列统计缺失值数量，返回延迟计算的Dask Series
        .compute()：触发实际计算，将分布式任务提交给Client管理的集群执行
        用各列缺失数除以总行数（ddf.shape[0]），计算结果四舍五入保留两位小数，生成缺失率百分比。
        '''

        # 生成结果报表
        result = dd.from_pandas(
            pd.DataFrame({
                'Missing_Count': missing_counts,
                'Missing_Ratio(%)': missing_ratio
            }).sort_values('Missing_Ratio(%)', ascending=False)  # 按缺失率排序[6](@ref)
            ,
            npartitions=1       # npartitions=1避免小数据分块带来的性能损耗
        )

        # 结果输出优化
        if self.manager.save:
            fmt = self.manager.setting.get('output_format', 'parquet')
            if fmt == 'parquet':
                result.to_parquet(
                    'missing_stats.parquet',
                    compression=self.manager.setting.get('compression', 'gzip'))
            elif fmt == 'csv':
                result.to_csv('missing_stats.csv', single_file=True)
        # 新增可视化调用
        if not self.manager.client._asynchronous:  # 非异步模式时执行
            self.plot_missing_rates(result)

        return result.compute()

class HandleStrategy:
    def __init__(self, manager: DataManager):
        self.manager = manager
        self.missing_ratio_dict = None
        self.strategy = {
            5:['drop', 'mean', 'median', 'mode'],
            20:['mi', 'knn', 'tree'],
            50:['multiple_imputation', 'binary_classification'],
            100: ['field_deletion', 'mi_verify', 'layered_filled'],
            'default': ['time','model']
        }

    def choose_model(self, table: pd.DataFrame):
        """根据各列缺失率创建字典映射

        参数:
            table: 包含缺失值统计结果的DataFrame，需包含'Missing_Ratio(%)'列

        其返回的引用可修改，请注意修改到只剩下唯一对应值
        """
        # 确保输入包含必要列
        if 'Missing_Ratio(%)' not in table.columns:
            raise ValueError("输入DataFrame必须包含'Missing_Ratio(%)'列")

        # 创建{列名: 缺失率}字典
        self.missing_ratio_dict = {
            row[0]: row[2]
            # for _, row in table.itertuples(index=True)  # 返回索引
            for row in table.itertuples(index=False)  # 使用命名元组方式遍历
            if row[2] > 0
        }


        self.manager.strategy = {
            column: list(self.strategy.get(self.clip_value(missing_ratio)))
            for column, missing_ratio in self.missing_ratio_dict.items()
        }

        return self.manager.strategy

    @staticmethod
    def clip_value(arr)->int:
        conditions = [
            arr <= 5,
            (arr > 5) & (arr <= 20),
            (arr > 20) & (arr <= 50),
            arr > 50
        ]
        choices = [5, 20, 50, 100]
        return np.select(conditions, choices).item()  # 确保返回标量


class HandleMissingValue:
    def __init__(self, manager: DataManager):
        self.manager = manager
        if not self.manager.ddf:
            self.manager.load_data()
        self.ddf = self.manager.ddf
        self.meta = self.ddf._meta

    def tag_encoding(self):
        """优化后的字符串转数值方法->标签编码"""
        # 全局统计所有类别
        str_cols = self.ddf.select_dtypes(include=['object']).columns
        categories = {col: self.ddf[col].astype('category').cat.categories.compute()
                      for col in str_cols}
        # 使用更高效的类别编码
        for col in str_cols:
            self.ddf[col] = self.ddf[col].map_partitions(
                lambda s: pd.Categorical(s, categories=categories[col]).codes,
                meta=(col, 'int32')
            )
        return self

    def one_hot_encoding(self):   # 后续可能改进点，使其适配动态数据（目前只适合静态数据，一次性统计完类别）
        """分布式友好的独热编码实现"""
        from sklearn.preprocessing import OneHotEncoder
        str_cols = self.ddf.select_dtypes(include=['object']).columns

        # 1. 全局统计所有类别
        global_categories = {}
        for col in str_cols:
            global_categories[col] = self.ddf[col].unique().compute().tolist()

        # 2. 初始化编码器（指定全局类别）
        encoder = OneHotEncoder(
            categories=[global_categories[col] for col in str_cols],
            sparse_output=False
        )
        encoder.fit([])  # 空拟合，仅用于生成特征名称

        # 3. 分区转换（仅调用 transform）
        encoded = self.ddf[str_cols].map_partitions(
            lambda df: encoder.transform(df),
            meta=pd.DataFrame(columns=encoder.get_feature_names_out(str_cols))
        )

        # 4. 合并结果
        self.ddf = dd.concat([self.ddf.drop(str_cols, axis=1), encoded], axis=1)
        return self


    def distribute(self)->pd.DataFrame:
        """优化后的分布式处理方法"""

        if not self.manager.strategy:
            raise ValueError("请先通过HandleStrategy设置处理策略")

        # 在处理流水线开始前添加
        self.ddf = self.ddf.persist()  # 将数据保持在各Worker内存中
        wait(self.ddf)  # 等待所有分区加载完成

        strategy_mapping = {
            'drop': self._process_drop,
            'mean': self._process_mean,
            'median': self._process_median,
            'mode': self._process_mode,
            'mi': self._process_mi,
            'knn': self._process_knn,
            'tree': self._process_tree,
            'multiple_imputation':self._process_multiple_imputation,
            'binary_classification': self._process_binary,
            'field_deletion': self._process_delete,
            'mi_verify':self._process_mi_verify,
            'layered_filled': self._process_layered,
            'time': self._process_time,  # 新增时间序列处理
            'model':self._process_model
        }

        # 按处理类型分组以提高效率
        from collections import defaultdict
        strategy_groups = defaultdict(list)
        for col, method in self.manager.strategy.items():
            strategy_groups[method].append(col)             # 生成同一策略下的列表字符串 cols（List[str]） 即使是仅有一个列也一样

        # 批量处理同类操作
        for method, cols in strategy_groups.items():
            if self.manager.smart_gc():  # 如果触发过回收，等待任务完成
                self.ddf = self.ddf.persist()  # 重新物化数据
                wait(self.ddf)  # 等待数据就绪

            if method in strategy_mapping:
                strategy_mapping[method](cols)
                self.manager.gc_counter +=1
            else:
                warnings.warn(f"未实现的方法: {method}")

            # 每处理self.manager.max_gc_counter组操作后强制检查
            if self.manager.gc_counter % self.manager.max_gc_counter == 0:
                self.manager.smart_gc(force=True)
                self.manager.gc_counter = 0

        # 保存处理结果
        if self.manager.save:
            self.ddf.to_parquet(
                'processed_data.parquet',
                compression=self.manager.setting.get('compression', 'gzip'),
                write_index=False
            )
        return self.ddf

    def _process_drop(self, columns:List[str]):
        """批量处理删除行"""
        self.ddf = self.ddf.dropna(subset=columns)

    def _process_mean(self, columns:List[str]):
        """批量均值填充"""
        means = self.ddf[columns].mean().compute()
        for col in columns:
            self.ddf[col] = self.ddf[col].fillna(means[col])

    def _process_median(self, columns:List[str]):
        """优化中位数计算"""
        medians = {}
        for col in columns:
            # 使用近似分位数计算
            medians[col] = self.ddf[col].quantile(0.5, method='tdigest').compute()
        for col in columns:
            self.ddf[col] = self.ddf[col].fillna(medians[col])

    def _process_mode(self, columns:List[str]):
        """优化众数计算"""
        for col in columns:
            mode = self.ddf[col].value_counts().idxmax().compute()
            self.ddf[col] = self.ddf[col].fillna(mode)

    def _process_mi(self, columns:List[str]):
        """修正后的多重插补方法"""
        imputer = SimpleImputer(strategy='mean')
        self.ddf[columns] = self.ddf[columns].map_partitions(
            lambda df: pd.DataFrame(
                imputer.fit_transform(df),
                columns=columns,
                index=df.index
            ),
            meta=self.meta[columns]
        )


    def _process_knn(self, columns:List[str]):
        """分布式优化的KNN缺失值填充"""
        # 客户端抽样（带容错机制）
        try:
            subsample = self.ddf[columns].sample(frac=0.1).compute().dropna()
        except ValueError:
            subsample = self.ddf[columns].head(1000, compute=True).dropna()

        # 广播子样本到所有Worker（带序列化优化）
        subsample_future = self.manager.client.scatter(subsample, broadcast=True)

        def knn_fill(df, subsample_):
            """Worker端的填充逻辑"""
            from sklearn.neighbors import KDTree

            if df[columns].isnull().any().any():
                # 重建KDTree（避免序列化对象）
                tree = KDTree(subsample_.values, leaf_size=40)

                # 定位缺失行
                missing_mask = df[columns].isnull().any(axis=1)
                missing_data = df.loc[missing_mask, columns]

                # 查询最近邻
                _, indices = tree.query(missing_data.values, k=5)
                imputed = np.nanmean(subsample_.iloc[indices], axis=1)

                # 原地更新
                df.loc[missing_mask, columns] = imputed
            return df

        # 应用分区处理（显式传递广播数据）
        self.ddf = self.ddf.map_partitions(
            knn_fill,
            subsample=subsample_future,
            meta=self.meta
        )

    def _process_tree(self, columns:List[str]):
        """批量树模型预处理"""
        self.ddf[columns] = self.ddf[columns].fillna(-999)
        # """树模型预处理"""
        # for col in columns:
        #     self.ddf[col] = self.ddf[col].fillna(-999)

    def _process_multiple_imputation(self, columns:List[str]):
        """多重插补+敏感性分析（并行化实现）"""
        imputer = ParallelPostFit(
            IterativeImputer(
                max_iter=10,
                random_state=42,
                estimator=RandomForestRegressor(n_estimators=10)
            )
        )

        # 转换为Dask数组
        dask_array = self.ddf[columns].to_dask_array(lengths=True)

        imputer.fit(dask_array)  # 先拟合
        imputed = imputer.transform(dask_array)  # 再转换

        # 转换回DataFrame
        self.ddf[columns] = dd.from_dask_array(
            imputed,
            columns=columns,
            meta=self.meta[columns]
        )

    def _process_binary(self, columns:List[str]):
        """批量生成缺失指示特征"""
        for col in columns:
            self.ddf[f"{col}_missing"] = self.ddf[col].isnull().astype(int)
            self.ddf[col] = self.ddf[col].fillna(self.ddf[col].mean())

    def _process_delete(self, columns:List[str]):
        """批量删除列"""
        self.ddf = self.ddf.drop(columns=columns)

    def _process_mi_verify(self, columns:List[str]):
        """优化后的分布式多重插补验证方法"""
        from dask_ml.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression

        # 分布式抽样（仅抽取5%数据用于验证）
        sample_ddf = self.ddf.sample(frac=0.05, random_state=42)

        # 分布式拆分训练集
        x_sample, y_sample = sample_ddf[columns], sample_ddf['label']
        x_train, x_test, y_train, y_test = train_test_split(
            x_sample, y_sample, test_size=0.2, shuffle=True
        )

        del x_sample, y_sample
        gc.collect()

        # 分布式多重插补
        imputed_datasets = []
        for i in range(3):
            imputer = ParallelPostFit(
                IterativeImputer(
                    max_iter=5,
                    random_state=i,
                    estimator=RandomForestRegressor(n_estimators=10)
                )
            )
            imputer.fit(x_train)
            imputed = imputer.transform(x_train)
            imputed_datasets.append(imputed)

        # 分布式模型训练
        results = []
        for X_imputed in imputed_datasets:
            model = ParallelPostFit(LogisticRegression())
            model.fit(X_imputed, y_train)
            score = model.score(x_test, y_test)
            results.append(score)  # 移除compute()

        # 应用最佳插补器
        best_idx = np.argmax(results)
        final_imputer = ParallelPostFit(
            IterativeImputer(
                max_iter=10,
                random_state=best_idx,
                estimator=RandomForestRegressor(n_estimators=20)
            )
        )
        final_imputer.fit(self.ddf[columns])
        self.ddf[columns] = final_imputer.transform(self.ddf[columns])

    def _process_layered(self, columns:List[str]):
        """优化分层填充（示例按月份）"""
        if 'date' not in self.ddf.columns:
            raise ValueError("需要日期列进行分层")

        self.ddf['month'] = self.ddf['date'].dt.month
        # grouped = self.ddf[columns + ['month']].groupby('month')

        # 使用Dask的map_partitions加速
        def fill_group(df, fill_cols):
            for col in fill_cols:
                group_means = df.groupby('month')[col].transform('mean')
                df[col] = df[col].fillna(group_means)
            return df.drop(columns='month')

        self.ddf = self.ddf.map_partitions(
            fill_group,
            fill_cols=columns,  # 使用重命名后的参数
            meta=self.meta
        )

    def _process_time(self, columns:List[str]):
        """时间序列专用处理方法"""
        time_col = self.manager.setting.get('time_column', 'date')
        if time_col not in self.ddf.columns:
            raise ValueError("时间列未配置，请在ManagerSettings中设置time_column")

        # 按时间排序
        self.ddf = self.ddf.set_index(time_col)

        for col in columns:
            self.ddf[col] = self.ddf[col].ffill()
            self.ddf[col] = self.ddf[col].map_partitions(
                # lambda s: s.interpolate(method='spline', order=3),    # 前向填充+三次样条插值组合
                lambda s: s.interpolate(method='time'),  # 关键修改点     # 前向填充+时间插值组合
                meta=(col, 'float64')
            )

        # 根据配置重分区
        if self.manager.setting.get('time_interval'):
            freq_map = {
                'd': '1D', 'm': '1M', 'y': '1Y',
                'h': '1H', 'min': '1T'
            }
            freq = freq_map.get(self.manager.setting['time_interval'], '1D')
            self.ddf = self.ddf.repartition(freq=freq)
        else:
            self.ddf = self.ddf.reset_index()

    def _process_model(self, columns: List[str]):
        """基于机器学习的预测插补（新增方法）"""
        from sklearn.ensemble import RandomForestRegressor   # 导入随机森林模型（非线性）
        from sklearn.linear_model import BayesianRidge       # 贝叶斯岭回归（线性）
        from dask_ml.wrappers import ParallelPostFit         # Dask的ParallelPostFit包装器实现并行预测

        for col in columns:
            # 创建临时数据集：使用其他列预测当前列
            temp_df = self.ddf.drop(col, axis=1)        # （axis=0 表示行，axis=1 表示列）
            target = self.ddf[col]

            # 获取原始索引（避免对其时乱序）
            original_index = self.ddf.index

            # 拆分存在和缺失的数据
            train_idx = target.notnull()        # 返回布尔值的Series
            test_idx = target.isnull()          # 返回布尔值的Series   返回延迟计算的Dask数组：[True, False, True, False]

            # 仅在存在缺失值时处理
            if test_idx.sum().compute() > 0:
                # 预处理：填充其他列的缺失值
                temp_df = temp_df.fillna(temp_df.mean())        # 按列填充各列的均值，若需计算行均值，需显式指定 axis=1（如 temp_df.mean(axis=1)）

                # 转换为Dask数组（保持分区）
                X_train = temp_df[train_idx].values     # 从Dask DataFrame中筛选出train_idx标记为True的行（即当前列非空的数据）
                y_train = target[train_idx].values      # .values：将筛选后的Dask DataFrame转换为Dask Array对象
                X_test = temp_df[test_idx].values

                # 使用并行包装器
                model = ParallelPostFit(
                    BayesianRidge() if X_train.shape[0] > 1e6 else  # 大数据用线性模型      shape[0]通常指的是第一个维度的大小，也就是样本数量（即行数）。
                    RandomForestRegressor(n_estimators=50, n_jobs=-1)
                )

                # 训练模型
                model.fit(X_train, y_train)

                # 预测缺失值
                preds = model.predict(X_test)

                # 显式设置预测值索引（保证生成的预测值和原本的缺失值索引对齐）
                missing_index = original_index[test_idx]  # 获取原始缺失位置索引
                pred_series = dd.from_dask_array(
                    preds,
                    columns=[col],
                    index=missing_index  # 关键修复：绑定原始缺失索引
                )

                # 更新原数据
                self.ddf[col] = dd.concat([
                    target[train_idx],  # 保留原始非空数据
                    pred_series         # 携带原始索引的预测值
                ], axis=0).repartition(divisions=self.ddf.divisions)  # 网页1：按原分区重组

                # 可选性能优化（网页2建议）
                if self.ddf.npartitions > 20:
                    self.ddf = self.ddf.persist()  # 网页2：内存持久化优化
                '''
                输入：
                    target[train_idx]：原始列中非空的已知值
                    preds：模型预测的缺失值
                输出：
                    合并后的完整列数据，并优化分区分布。
                '''
# 使用示例
"""
if __name__ == "__main__":
    # 配置参数（包含时间序列配置）
    config = ManagerSettings(
        path="large_dataset.parquet",
        blocksize="256MB",
        na_value=["NA", "null"],
        dtype={'timestamp': 'datetime64[ns]'},
        compression="snappy"
        output_format="parquet",
        time_column="timestamp",  # 时间列配置
        time = None             # 默认填None，是时间序列间隔的时间 min,h,d,m,y
    )

    # 初始化管理器（显式优化配置）
    DataManager.optimize_dask_config()  # 全局配置优化
    manager = DataManager(
        config, 
        save=True,
        n_workers=8,  # 根据CPU核心数调整
        memory_limit='4GB'  # 根据物理内存调整
    )

    # 执行分析流程
    analyzer = MissingValueAnalyzer(manager)
    result = analyzer.analyze_missing_values()
    
    # 策略选择与处理
    strategy_selector = HandleStrategy(manager)
    strategy_table = strategy_selector.choose_model(result)
    
    manager.select_strategy()  # <- 新增的选择策略步骤
    
    processor = HandleMissingValue(manager)
    
    # 需要显式调用编码类型（二选一）
    processor.tag_encoding()        # 标签编码
    processor.one_hot_encoding()    # 独热编码
    
    processed_data = processor.distribute()
    
    # 结果输出
    if manager.save:
        processed_data.compute().to_parquet("final_result.parquet")
"""