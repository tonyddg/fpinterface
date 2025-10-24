# 安装 foundation pose
cd /app/FoundationPose
bash ./build_all.sh

# 安装必要的模块
pip install -r /app/requirements.txt

# 将 /app/FoundationPose/estimater.py 中的代码修改为相对引用
sed -i \
    -e 's/^from Utils import \*/from .Utils import */' \
    -e 's/^from datareader import \*/from .datareader import */' \
    -e 's/^from learning\.training\.predict_score import \*/from .learning.training.predict_score import */' \
    -e 's/^from learning\.training\.predict_pose_refine import \*/from .learning.training.predict_pose_refine import */' \
    "/app/FoundationPose/estimater.py"