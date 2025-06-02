# Data_preprocessing-数据预处理
两个方便的类，分别用于处理数据的缺失值和异常值  
DataLack: 负责处理缺失值  
Outliers:负责处理异常值  
Two convenient classes for handling missing and outliers in your data  
DataLack: Responsible for handling missing values  
Outliers: Responsible for handling outliers  

# JudgeMethod流程图
graph TD  
    A[judge_distribution] --> B{数值型?}  
    B -->|否| C[返回 categorical]  
    B -->|是| D[计算nunique/total]  
    D --> E{离散型条件?}  
    E -->|是| F[_judge_discrete]  
    E -->|否| G[_judge_continuous]  
    G --> H[计算偏度/峰度]  
    H --> I[分层抽样]  
    I --> J[正态性检验]  
    J --> K{是否正态?}  
    K -->|是| L[返回normal]  
    K -->|否| M{是否复杂分布?}  
    M -->|是| N[拟合分布+形状检测]  
    M -->|否| O[返回矩量判断结果]  
    F --> P{唯一值≤2?}  
    P -->|是| Q[二项分布检验]  
    P -->|否| R[泊松/负二项检验]  
    Q & R --> S[返回离散型分布]  

