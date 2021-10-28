import cv2
import numpy as np
import os
import decimal
import webcolors
import extcolors
import PIL
from PIL import Image
import json

import easyocr
reader = easyocr.Reader(['de', 'en'])


# https://stackoverflow.com/questions/61512970/how-to-find-only-the-bolded-text-lines-from-no-of-images
def get_bold_style_list(dataframe, image):

    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5,5),np.uint8)
    kernel2 = np.ones((3,3),np.uint8)
    marker = cv2.dilate(thresh,kernel,iterations=1)
    mask = cv2.erode(thresh,kernel,iterations=1)

    while True:
        tmp = marker.copy()
        marker = cv2.erode(marker, kernel2)
        marker = cv2.max(mask, marker)
        difference = cv2.subtract(tmp, marker)
        if cv2.countNonZero(difference) == 0:
            break

    marker_color = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    out = cv2.bitwise_or(img, marker_color)
    cv2.imwrite('./data/crop/bold_img.png', out)

    # Doing OCR. Get bounding boxes.
    bounds_style = reader.readtext('./data/crop/bold_img.png')

    bold_list = [0] * dataframe.shape[0]
    for src_idx, src_row in dataframe.iterrows():
        df_xmin_1 = src_row['x_min_1']
        df_xmin_2 = src_row['x_min_2']
        df_xmax_1 = src_row['x_max_1']
        df_xmax_2 = src_row['x_max_2']
        df_ymax_1 = src_row['y_max_1']
        df_ymax_2 = src_row['y_max_2']
        df_ymin_1 = src_row['y_min_1']
        df_ymin_2 = src_row['y_min_2']

        for i in range(len(bounds_style)):
            coordinates = bounds_style[i][0]
            text = bounds_style[i][1]
            #             print(coordinates, text)
            xmin_1 = coordinates[0][0]
            xmin_2 = coordinates[0][1]
            xmax_1 = coordinates[1][0]
            xmax_2 = coordinates[1][1]
            ymax_1 = coordinates[2][0]
            ymax_2 = coordinates[2][1]
            ymin_1 = coordinates[3][0]
            ymin_2 = coordinates[3][1]

            if ((int(df_xmin_1) >= int(xmin_1 ) -10 and int(df_xmin_1) <= int(xmin_1 ) +10) and
                    (int(df_xmin_2) >= int(xmin_2 ) -10 and int(df_xmin_2) <= int(xmin_2 ) +10) and
                    (int(df_xmax_1) >= int(xmax_1 ) -10 and int(df_xmax_1) <= int(xmax_1 ) +10) and
                    (int(df_xmax_2) >= int(xmax_2 ) -10 and int(df_xmax_2) <= int(xmax_2 ) +10) and
                    (int(df_ymax_1) >= int(ymax_1 ) -10 and int(df_ymax_1) <= int(ymax_1 ) +10) and
                    (int(df_ymax_2) >= int(ymax_2 ) -10 and int(df_ymax_2) <= int(ymax_2 ) +10) and
                    (int(df_ymin_1) >= int(ymin_1 ) -10 and int(df_ymin_1) <= int(ymin_1 ) +10) and
                    (int(df_ymin_2) >= int(ymin_2 ) -10 and int(df_ymin_2) <= int(ymin_2 ) +10)):

                bold_list[src_idx] = 1

    if os.path.exists('./data/crop/bold_img.png'):
        os.remove('./data/crop/bold_img.png')

    return bold_list


# https://stackoverflow.com/questions/62669589/extract-text-with-strikethrough-from-image
def get_underline_list(dataframe, image):
    img = cv2.imread(image)
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((4, 2), np.uint8)
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    trans = dilation

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
    detect_horizontal = cv2.morphologyEx(trans, cv2.MORPH_OPEN, horizontal_kernel, iterations=5)
    cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(result, [c], -1, (36, 255, 12), 2)

    underline_list = [0] * dataframe.shape[0]
    for c in cnts:
        try:
            p0, p1, p2 = c[0:3]
            underline_len = abs(p2[0][0] - p0[0][0])

            for src_idx, src_row in dataframe.iterrows():
                df_xmin_1 = src_row['x_min_1']
                df_xmax_1 = src_row['x_max_1']

                if (abs(p0[0][0] - int(df_xmin_1)) <= 15) and (abs(p2[0][0] - int(df_xmax_1)) <= 15) and \
                        abs(abs(int(df_xmax_1) - int(df_xmin_1)) - underline_len) <= 15:
                    underline_list[src_idx] = 1

        except Exception as e:
            pass

    return underline_list


# https://www.geeksforgeeks.org/python-find-most-frequent-element-in-a-list/
def most_frequent(List):
    dict = {}
    count, itm = 0, ''
    for item in reversed(List):
        dict[item] = dict.get(item, 0) + 1
        if dict[item] >= count:
            count, itm = dict[item], item
    return (itm)


# https://stackoverflow.com/questions/9694165/convert-rgb-color-to-english-color-name-like-green-with-python
def closest_colour(requested_colour):
    min_colours = {}
    colors_dict = webcolors.CSS3_HEX_TO_NAMES
    for hash_key in colors_dict:
        r_c, g_c, b_c = webcolors.hex_to_rgb(hash_key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = colors_dict[hash_key]

    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def get_colors_list(dataframe, image, crop_loc='./data/crop/'):
    im = PIL.Image.open(image)

    for src_idx, src_row in dataframe.iterrows():
        df_xmin_1 = src_row['x_min_1']
        df_xmin_2 = src_row['x_min_2']
        df_ymax_1 = src_row['y_max_1']
        df_ymax_2 = src_row['y_max_2']

        x_min = min(int(df_xmin_1), int(df_ymax_1))
        y_min = min(int(df_xmin_2), int(df_ymax_2))
        x_max = max(int(df_xmin_1), int(df_ymax_1))
        y_max = max(int(df_xmin_2), int(df_ymax_2))

        # to crop the images
        file_coordinates = (x_min, y_min, x_max, y_max)
        area = im.crop(file_coordinates)
        filename = crop_loc + str(file_coordinates)
        if not os.path.exists(filename):
            filename2 = filename + '.jpg'
            area.save(filename2)
            # end of cropping

    color_list = []
    for filename_ in os.listdir(crop_loc):
        try:
            filename1 = crop_loc + filename_
            if os.path.exists(filename1):
                i = Image.open(filename1)
                colors, pixel_count = extcolors.extract_from_image(i)
                actual_name, closest_name = get_colour_name(colors[1][0])
                if 'gray' in closest_name:
                    closest_name = 'gray'

                color_list.append(closest_name)
                os.remove(filename1)

        except Exception as e:
            pass

    most_freq_color = most_frequent(color_list)

    if len(color_list) < dataframe.shape[0]:
        while len(color_list) < dataframe.shape[0]:
            color_list.append(most_freq_color)

    return color_list


def get_ner_list(dataframe, json_filename, final_df):
    ner_list = ['other'] * final_df.shape[0]


    # Opening JSON file
    with open(json_filename) as json_file:
        data_dict = json.load(json_file)
        for key in data_dict:
            if data_dict[key] != '':
                for src_idx, src_row in dataframe.iterrows():
                    df_content = src_row['content']
                    df_content = df_content.replace('\,', ',')

                    if (data_dict[key].lower().strip() == df_content.lower().strip()) or (
                            data_dict[key].lower().strip() in df_content.lower().strip()):
                        if key == 'total':
                            key1 = data_dict[key]
                            key2 = df_content

                            if 'RM' in key1:
                                key1 = key1.strip()
                                key1 = key1.replace('RM', '')
                            if 'RM' in key2:
                                key2 = key2.strip()
                                key2 = key2.replace('RM', '')

                            if isinstance(key1, float) and isinstance(key2, float):
                                d = decimal.Decimal(key1)
                                d1 = decimal.Decimal(key2)
                                print(d, d1, d.as_tuple().exponent, d1.as_tuple().exponent)
                                if int(abs(d.as_tuple().exponent)) == int(abs(d1.as_tuple().exponent)):
                                    ner_list[src_idx] = key
                                    break
                        else:
                            ner_list[src_idx] = key
                            break

                    final_match_str = df_content.lower().strip().replace(', ', ',')
                    final_match_str1 = data_dict[key].lower().strip().replace(', ', ',')
                    if final_match_str in final_match_str1 and not final_match_str.isnumeric():
                        ner_list[src_idx] = key

    return ner_list