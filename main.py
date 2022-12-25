import io
import os
import cv2
from PIL import Image
import PySimpleGUI as sg


image = Image.open(r'test.jpg')
image.thumbnail((200, 400))
bio = io.BytesIO()
image.save(bio, format="PNG")

layout = [
    [sg.Image(key="-IMAGE-")],
    [sg.Button("COMPARAR", size=(24, 2), mouseover_colors='gray')],
    [sg.Image(data=None, key="img_display")],
    [sg.Text('', key='mensagem')]

]

window = sg.Window("Image Viewer", layout, finalize=True)
window["-IMAGE-"].update(data=bio.getvalue())


while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED:
        break

    if event == "COMPARAR":

        source_image = cv2.imread("test.jpg")
        score = 0
        file_name = None
        image = None
        kp1, kp2, mp = None, None, None

        for file in [file for file in os.listdir("database")][:3]:
            target_image = cv2.imread("./database/" + file)

            sift = cv2.SIFT.create()
            kp1, des1 = sift.detectAndCompute(source_image, None)
            kp2, des2 = sift.detectAndCompute(target_image, None)
            matches = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=10), dict()).knnMatch(des1, des2, k=2)

            mp = []
            for p, q in matches:
                if p.distance < 0.1 * q.distance:
                    mp.append(p)
                    keypoints = 0
                if len(kp1) <= len(kp2):
                    keypoints = len(kp1)
                else:
                    keypoints = len(kp2)

                if len(mp) / keypoints * 100 > score:
                    score = len(mp) / keypoints * 100
                    print('The best match :' + file)
                    print('The score :' + str(score))
                    result = cv2.drawMatches(
                        source_image, kp1, target_image, kp2, mp, None)
                    result = cv2.resize(result, None, fx=1.1, fy=1.1)
                    cv2.imshow("result", result)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    break
