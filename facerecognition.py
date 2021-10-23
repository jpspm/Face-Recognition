from PIL.Image import new
import numpy as np
import face_recognition as fr
import cv2
from numpy.core.numeric import True_

known_face_encondings = []
known_face_names = []

def newPerson():
    nome = input("Qual o nome do CLiente: ")
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "{}.png".format(nome)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            break

    cam.release()

    cv2.destroyAllWindows()
    newClient = fr.load_image_file("{}.png".format(nome))
    newClient_face_encoding = fr.face_encodings(newClient)[0]
    known_face_encondings.append(newClient_face_encoding)
    known_face_names.append(nome)

def loadPerson():
    video_capture = cv2.VideoCapture(0)
    while True: 
        ret, frame = video_capture.read()

        rgb_frame = frame[:, :, ::-1]

        face_locations = fr.face_locations(rgb_frame)
        face_encodings = fr.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = fr.compare_faces(known_face_encondings, face_encoding)

            name = "Unknown"

            face_distances = fr.face_distance(known_face_encondings, face_encoding)

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        cv2.imshow('Webcam_facerecognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def main():
    while True:
        print("1 - Novo Cliente")
        print("2 - Cliente já cadastrado")
        print("3 - Sair")
        op = int(input("Digite a opção: "))
        if op == 1:
            newPerson()
        elif op == 2:
            loadPerson()
        elif op == 3:
            break
        else:
            print("Opção inválida!")

main()