import cv2
import sys
import numpy as np
import datetime
import os
import glob
from RetinaFace.retinaface import RetinaFace
from PIL import Image
from scipy import misc
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face


def align(landmarks, img):

    refrence = get_reference_facial_points(default_square=True)
    # facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
    facial5points = landmarks[0][:][:]
    warped_face = warp_and_crop_face(np.array(img), facial5points, refrence, crop_size=(112, 112))
    return warped_face



def list_img_name(path,img_ext):
    img_list = []
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path, folder_name)
        for img_name in os.listdir(folder_path):
            if os.path.splitext(img_name)[-1] == img_ext:
                img_path = os.path.join(folder_path, img_name)
                img_list.append(img_path)
    return img_list

thresh = 0.8
scales = [1024, 1980]

count = 1

gpuid = 0
detector = RetinaFace('/home/yeluyue/dl/Datasets/ICCV_challenge/insightface-master/models/retinaface-R50/R50', 0, gpuid, 'net3')

dataset_path = '/home/yeluyue/yeluyue/DataSet/lfw/all/lfw_sor/val'
img_ext = '.jpg'
imgs_list = list_img_name(dataset_path, img_ext)
imgs_list = sorted(imgs_list)

dataset_path_save = '/home/yeluyue/yeluyue/DataSet/lfw/Patches_retinaface_112x112/lfw_retinaface_align'
people_num = 0
img_num = 0
nrof_successfully_aligned = 0
for img_path in imgs_list:

    img_save_folder = os.path.join(dataset_path_save, img_path.rsplit('/')[-2])
    if not os.path.exists(img_save_folder):
        os.mkdir(img_save_folder)
        people_num += 1
    img_name_jpg = img_path.rsplit('/')[-1]
    img_name_png = img_name_jpg.replace(".jpg", ".png")
    img_path_save = os.path.join(img_save_folder, img_name_png)
    scales = [1024, 1980]
    img = cv2.imread(img_path)
    # print(img.shape)
    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    # im_scale = 1.0
    # if im_size_min>target_size or im_size_max>max_size:
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    # print('im_scale', im_scale)

    scales = [im_scale]
    flip = False

    for c in range(count):
        faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
        bounding_boxes = faces
        # print(c, faces.shape, landmarks.shape)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 1:
        print(img_path_save)
        print('find', faces.shape[0], 'faces')
        length = 0
        width = 0
        det = np.zeros(4, dtype=np.int32)
        landmarks_use = np.zeros([1, 5, 2], dtype=np.int32)
        for i in range(faces.shape[0]):
            length_1 = faces[i][2] - faces[i][0]
            width_1 = faces[i][3] - faces[i][1]
            if length_1 >= length and width_1 >= width:
                length = length_1
                width = width_1
                det = faces[i, 0:4]
                landmarks_use[0][:][:] = landmarks[i][:][:]
    else:
        det = np.zeros(4, dtype=np.int32)
        det = faces[0, 0:4]
        landmarks_use = np.zeros([5, 2], dtype=np.int32)
        landmarks_use = landmarks


    img_align = align(landmarks_use, img)
    cv2.imwrite(img_path_save, img_align)

    # img_align.save(img_path_save)

    # margin = 20
    # img_size = np.asarray(img.shape)[0:2]
    # bb = np.zeros(4, dtype=np.int32)
    # bb[0] = np.maximum(det[0] - margin / 2, 0)
    # bb[1] = np.maximum(det[1] - margin / 2, 0)
    # bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    # bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    # cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    # scaled = misc.imresize(cropped, (112, 112), interp='bilinear')
    # nrof_successfully_aligned += 1
    # cv2.imwrite(img_path_save, scaled)



    # if nrof_faces > 0:
    #     det = bounding_boxes[:, 0:4]
    #     img_size = np.asarray(img.shape)[0:2]
    #     if nrof_faces > 1:
    #         bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
    #         img_center = img_size / 2
    #         offsets = np.vstack(
    #             [(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
    #         offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
    #         index = np.argmax(bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
    #         det = det[index, :]
    #     det = np.squeeze(det)
    #     bb = np.zeros(4, dtype=np.int32)
    #     margin = 20
    #     bb[0] = np.maximum(det[0] - margin / 2, 0)
    #     bb[1] = np.maximum(det[1] - margin / 2, 0)
    #     bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
    #     bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
    #     cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
    #     scaled = misc.imresize(cropped, (112, 112), interp='bilinear')
    #     nrof_successfully_aligned += 1
    #     cv2.imwrite(img_path_save, scaled)
    #     # text_file.write('%s %d %d %d %d\n' % (output_filename, bb[0], bb[1], bb[2], bb[3]))
    # else:
    #     print('Unable to align "%s"' % img_path_save)

    # if faces is not None:
    #     print('find', faces.shape[0], 'faces')
    #     for i in range(faces.shape[0]):
    #         # print('score', faces[i][4])
    #         box = faces[i].astype(np.int)
    #         # color = (255,0,0)
    #         color = (0, 0, 255)
    #         cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
    #         if landmarks is not None:
    #             landmark5 = landmarks[i].astype(np.int)
    #             # print(landmark.shape)
    #             for l in range(landmark5.shape[0]):
    #                 color = (0, 0, 255)
    #                 if l == 0 or l == 3:
    #                     color = (0, 255, 0)
    #                 cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    # img.save(img_path_save)
    # cv2.imwrite(img_path_save, img)
    img_num += 1
    print('img_num:', img_num)
print('peolpe_num:', people_num)




