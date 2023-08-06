import re
import shutil

String = 'mAP: 0.89 Done'
mAP = re.findall(r"(?<=mAP: )\d+\.?\d*", String)[0]
print(type(mAP))
mAP = float(mAP)
print(type(mAP))
print(mAP)

# path = './checkpoints/hku_mmdetector_best.pth'
# save_path = './best_model'
#
# shutil.copy(path, save_path)

print("Best mAP:{:.2f}".format(0.44))
print("Best parameters:{}".format({'lr': 5e-5, 'bs': 8}))