# switchtransfomer
数据来源 : https://www.statmt.org/wmt14/translation-task.html  
运行环境 : python3.10.1 , PyTorch: 2.9.1+cu128 , RTX 5090*8
有一个需求 : cuda的序号必须从0开始并且连续。  
Data的格式为,这个文件夹需要按照这个格式创建，下载的训练数据放在RaWTrainData中，下载的评估数据放在RawTestData中  
Data/  
├── .pt/  
├── RawData/  
│   ├── RawTestData/  
│   └── RawTrainData/  
├── Data/  
│   ├── TestData/  
│   └── TrainData/  
这个的实现相较原版transfomer有点复杂，主要是每个进程的专家得少，总共16个专家，每个进程得独立创建两个专家，这两个转变不能加入了DDP中，得分开，模型保存的时候也是得分开保存。但是共享参数部分需要用DDP包装。  
还有一个问题就是要处理不同的模块所需的学习率会不一样，这一点和transfomer原版不同，路由器因为关键，而且信息多，所以学习率和裁剪都很小，但是专家不一样，在负载均衡的情况下分到的数据只有batch_size/16所以
为了缓解这个问题就只能把专家参数对应的学习率，剪裁都要调大一些
最终实现的效果(d_model=512,en-fr)    
#Greedy评估 : transfomer_512_en_fr_20000.pt  
BLEU = 32.55 62.7/39.0/26.2/17.9 (BP = 0.994 ratio = 0.994 hyp_len = 76820 ref_len = 77306)  
BLEU = 32.545667072655625  
#Greedy评估 : transfomer_512_en_fr_40000.pt  
BLEU = 33.77 64.1/40.7/27.7/19.2 (BP = 0.983 ratio = 0.983 hyp_len = 76001 ref_len = 77306)  
BLEU = 33.77229341510525  
#Greedy评估 : transfomer_512_en_fr_60000.pt  
BLEU = 34.72 64.0/40.9/28.2/19.7 (BP = 1.000 ratio = 1.003 hyp_len = 77526 ref_len = 77306)  
BLEU = 34.715434048262296  
#Greedy评估 : transfomer_512_en_fr_80000.pt  
BLEU = 35.61 64.9/42.0/29.2/20.6 (BP = 0.995 ratio = 0.995 hyp_len = 76913 ref_len = 77306)  
BLEU = 35.611263257372954  
#Greedy评估 : transfomer_512_en_fr_100000.pt  
BLEU = 35.68 65.2/42.2/29.3/20.7 (BP = 0.993 ratio = 0.993 hyp_len = 76735 ref_len = 77306)  
BLEU = 35.67591066226406  
