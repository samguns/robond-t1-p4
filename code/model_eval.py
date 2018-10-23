import tensorflow as tf
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils import model_tools
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D

weight_file_name = 'model_weights'
model = model_tools.load_network(weight_file_name)
#weights_path = "../data/weights/weights.28.h5"
#model.load_weights(weights_path)

run_num = 'run_1'

val_with_targ, pred_with_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_with_targ', 'sample_evaluation_data')
val_no_targ, pred_no_targ = model_tools.write_predictions_grade_set(model,
                                        run_num,'patrol_non_targ', 'sample_evaluation_data')
val_following, pred_following = model_tools.write_predictions_grade_set(model,
                                        run_num,'following_images', 'sample_evaluation_data')

true_pos1, false_pos1, false_neg1, iou1 = scoring_utils.score_run_iou(val_following, pred_following)
true_pos2, false_pos2, false_neg2, iou2 = scoring_utils.score_run_iou(val_no_targ, pred_no_targ)
true_pos3, false_pos3, false_neg3, iou3 = scoring_utils.score_run_iou(val_with_targ, pred_with_targ)

true_pos = true_pos1 + true_pos2 + true_pos3
false_pos = false_pos1 + false_pos2 + false_pos3
false_neg = false_neg1 + false_neg2 + false_neg3

weight = true_pos/(true_pos+false_neg+false_pos)
print(weight)

final_IoU = (iou1 + iou3)/2
print(final_IoU)

final_score = final_IoU * weight
print(final_score)