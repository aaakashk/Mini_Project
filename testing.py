import cv2
from keras.models import load_model
import numpy as np

def alphabet_dict():
    my_dict = {key - 65: chr(key) for key in range(ord('A'), ord('Z') + 1)}
    return my_dict

def image_preprocessor(img):
    img = cv2.GaussianBlur(img, (7,7), 0)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)
    final_img = cv2.resize(thresh_img, (28,28))
    final_img = np.reshape(final_img, (1,28,28,1))
    return final_img

def main():
    eng_dict = alphabet_dict()
    loaded_model = load_model('/Users/akashkumar/Codes/Python/Mini_Project/model_hand_char.h5')

    for i in range(65, 90+1):
        path = f'/Users/akashkumar/Documents/images/{i}.png'
        img = cv2.imread(path)
        copied_img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))

        processed_img = image_preprocessor(copied_img)

        label_pred = np.argmax(loaded_model.predict(processed_img))
        predicted_img = eng_dict[label_pred]

        char_i = chr(i)
        window_name = f'Image {char_i}'
        text_position = (10,280)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 1
        font_color = (252, 3, 165)
        thickness = 1
        line_type = 2

        cv2.putText(img,
                    'Prediction: ' + predicted_img,
                    text_position,
                    font,
                    font_size,
                    font_color,
                    thickness,
                    line_type)
        cv2.imshow(window_name, img)
        while True:
            key = cv2.waitKey(0) & 0xFF
            #27 == ASCII code of esc key
            #13 == ASCII code of carriage return or 'Enter'
            if key == 27 or key == 13 or key == ord('q') or key == ord('Q'):
                break
            if key == ord('S') or key == ord('s'):
                new_file_name = f'Predicted image of {char_i}.png'
                cv2.imwrite(new_file_name, img)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()