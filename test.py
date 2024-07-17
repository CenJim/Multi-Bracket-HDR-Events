import numpy as np
import cv2
from utils.preprocessing import read_timestamps_from_file
from utils.vision_quality_compare import calculate_psnr
from utils.load_hdf import get_dataset_shape

def check_npy(data_path, data_type: str = 'npy'):
    data = []
    if data_type == 'npy':
        data = np.load(data_path)
    print(data.shape)
    with np.load(data_path) as data:
        # print(data.shape)
        for key in data:
            # 获取数组
            array = data[key]
            # nonzero = np.count_nonzero(array[1][2])
            print(f"Array under key '{key}':\n{array.shape}\n")

    timestamps_path = '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_c/Interlaken Exposure Left.txt'
    timestamps_pair = read_timestamps_from_file(timestamps_path)
    print(f'the average exposure time is: {np.mean([int(pair[1]) - int(pair[0]) for pair in timestamps_pair])}')


def npy_to_image(data_path):
    img = np.load(data_path)
    img = np.transpose(img, (1, 2, 0))
    print(img.shape)
    print(img[:, :, 0:3])
    # cv2.imshow('Processed Image', cv2.cvtColor(img[:, :, 0:3], cv2.COLOR_RGB2BGR))
    cv2.imwrite('temp/hdr.png', cv2.cvtColor((img[:, :, 0:3] * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    # print('press \'q\' or \'Esc\' to quit')
    # k = cv2.waitKey(0)
    # if k == 27 or k == ord('q'):  # 按下 ESC(27) 或 'q' 退出
    #     cv2.destroyAllWindows()


if __name__ == '__main__':
    # img_path = ''
    # correct_img_path = ''
    # calculate_psnr(img_path, correct_img_path)
    print(get_dataset_shape('/home/s2491540/dataset/DSEC/train/interlaken_00_c/Interlaken_events_left'))