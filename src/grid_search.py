import os
import re
import shutil

best_map = 0
best_parameters = {}
path = './checkpoints/hku_mmdetector_best.pth'
save_path = './best_model'


for lr in [1e-4, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4]:
    for bs in [13, 7, 8, 9, 10, 11, 12, 13]:
        os.system('python train.py --output_dir \'checkpoints\' --batch_size %d --learning_rate %f' % (bs, lr))
        line = os.popen('python eval.py --split \'val\' --output_file "./result.pkl"').read()
        mAP = re.findall(r"(?<=mAP: )\d+\.?\d*", line)[0]
        mAP = float(mAP)
        print(mAP)
        if mAP > best_map:  # 找到表现最好的参数
            best_map = mAP
            shutil.copy(path, save_path)
            best_parameters = {'lr': lr, 'bs': bs}

print("Best mAP:{:.2f}".format(best_map))
print("Best parameters:{}".format(best_parameters))
