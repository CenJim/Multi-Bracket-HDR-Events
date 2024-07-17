import numpy as np
import cv2
from utils.preprocessing import read_timestamps_from_file

if __name__ == '__main__':
    # img = np.load(
    #     '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_c/Interlaken Left Images processed/000001_2.npy')
    # img = np.load(
    #     '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_c/Interlaken Left Images supervised/000001.npy')
    # print(img.shape)
    # print(img[:, :, 0:3])
    # # cv2.imshow('Processed Image', cv2.cvtColor(img[:, :, 0:3], cv2.COLOR_RGB2BGR))
    # cv2.imwrite('temp/image_reference.png', cv2.cvtColor((img[:, :, 0:3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # print('press \'q\' or \'Esc\' to quit')
    # k = cv2.waitKey(0)
    # if k == 27 or k == ord('q'):  # 按下 ESC(27) 或 'q' 退出
    #     cv2.destroyAllWindows()

    data = np.load('/Volumes/CenJim/train data/dataset/DSEC/train/train_sequences/sequence_0000000/hdr_images/000001.npy')
    print(data.shape)
    # with np.load(
    #     '/Volumes/CenJim/train data/dataset/DSEC/train/train_sequences/sequence_0000000/events/000000_1.npz') as data:
    #     # print(data.shape)
    #     for key in data:
    #         # 获取数组
    #         array = data[key]
    #         # nonzero = np.count_nonzero(array[1][2])
    #         print(f"Array under key '{key}':\n{array.shape}\n")

    # timestamps_path = '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_c/Interlaken Exposure Left.txt'
    # timestamps_pair = read_timestamps_from_file(timestamps_path)
    # print(f'the average exposure time is: {np.mean([int(pair[1]) - int(pair[0]) for pair in timestamps_pair])}')
