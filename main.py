import cv2
import streamlit as st
import mediapipe as mp
import math as m
import pandas as pd
import numpy as np
import tempfile
import matplotlib.pyplot as plt

DEMO_VIDEO = './horizonte.mp4'

# Inicialize o mediapipe
mp_pose = mp.solutions.pose

# Inicialize o Streamlit
st.title('Analisador Postural')
st.sidebar.title('Protótipo versão streamlit')
st.sidebar.subheader('Parametros')

# Botão para usar a webcam
use_webcam = st.sidebar.checkbox('Use Webcam')

# Carregar arquivo de vídeo
video_file_buffer = None
video_file = st.sidebar.file_uploader("Upload um video aqui", type=["mp4", "mov", "avi"])

if video_file is not None:
    video_file_buffer = tempfile.NamedTemporaryFile(delete=False)
    video_file_buffer.write(video_file.read())
    file_name = video_file_buffer.name
else:
    if use_webcam:
        file_name = 0
    else:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        file_name = DEMO_VIDEO  # Substitua pelo nome do vídeo padrão


st.sidebar.video(file_name)
# Threshold para alinhamento dos ombros
alignment_threshold = st.sidebar.slider('Defina o nivel de confiabilidade (Threshold)', min_value=0, max_value=100, value=50)

# Gráfico de barras
st.subheader('Resultado das Analises')
chart = st.empty()

# Função para calcular a distância
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Função para calcular o ângulo
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

# Função para enviar um aviso (substitua pelo que for necessário)
def sendWarning(x):
    pass

# Cores


good_frames = 0
bad_frames = 0

font = cv2.FONT_HERSHEY_SIMPLEX

blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Inicializar mediapipe
pose = mp_pose.Pose()

# Inicializar captura de vídeo
cap = cv2.VideoCapture(0) if use_webcam else cv2.VideoCapture(file_name)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_output = cv2.VideoWriter('./output2.mp4', fourcc, fps, frame_size)

print('Processing..')
stframe = st.empty()
output_ang = st.empty()
output_ali = st.empty()
alitotal = []
aliframe = 0 
desaliframe = 0
angbracos = []
boaPostura = []
maPostura = []
angbracos = []
good_frames = 0
bad_frames = 0

while cap.isOpened():
    success, image = cap.read()
    
    if not success:
        break

    h, w = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    if lm is not None:
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # Obtém as coordenadas das articulações do ombro, cotovelo e mão
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

        l_elbow_x = int(lm.landmark[lmPose.LEFT_ELBOW].x * w)
        l_elbow_y = int(lm.landmark[lmPose.LEFT_ELBOW].y * h)

        l_wrist_x = int(lm.landmark[lmPose.LEFT_WRIST].x * w)
        l_wrist_y = int(lm.landmark[lmPose.LEFT_WRIST].y * h)

# Calcula os vetores entre o ombro, cotovelo e mão
        shoulder_to_elbow = (l_elbow_x - l_shldr_x, l_elbow_y - l_shldr_y)
        elbow_to_wrist = (l_wrist_x - l_elbow_x, l_wrist_y - l_elbow_y)

# Calcula o ângulo entre os vetores usando a função atan2
        angle_rad = m.atan2(elbow_to_wrist[1], elbow_to_wrist[0]) - m.atan2(shoulder_to_elbow[1], shoulder_to_elbow[0])

# Converte o ângulo de radianos para graus
        angle_deg = m.degrees(angle_rad)
        # Garante que o ângulo seja positivo
        angle_deg = abs(angle_deg)

# Limita o ângulo a no máximo 90 graus
        angle_deg = min(angle_deg, 90)
        angbracos.append(angle_deg) 
        media_bracosAng = np.mean(angbracos, dtype=np.float64)
        
        output_ang.text(f"## Ângulo dos Braços: {media_bracosAng:.2f} graus")
        
# Agora, você tem o ângulo do braço em graus
        #print(f"Ângulo do braço esquerdo: {angle_deg} graus")


        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)
        
        if offset < alignment_threshold:
            #st.sidebar.text('Alignment: Aligned')
            aliframe += 1

            
        else:
            #st.sidebar.text('Alignment: Not Aligned')
            desaliframe += 1 
        
        # Resto do seu código para cálculo de ângulo
        # ...
        
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)


        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)


        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)


        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        angle_text_string = 'Cevical : ' + str(int(neck_inclination)) + '  Tronco : ' + str(int(torso_inclination))


        if neck_inclination < 40 and torso_inclination < 10:
            bad_frames = 0
            good_frames += 1

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)


            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

        else:
            good_frames = 0
            bad_frames += 1

            cv2.putText(image, angle_text_string, (10, 30), font, 0.9, red, 2)
            cv2.putText(image, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
            cv2.putText(image, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)


            cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
            cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
            cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)


        good_time = (1 / fps) * good_frames
        bad_time =  (1 / fps) * bad_frames

        if good_time > 0:
            time_string_good = 'Tempo em Boa Postura : ' + str(round(good_time, 1)) + 's'
            cv2.putText(image, time_string_good, (10, h - 20), font, 0.9, green, 2)
        else:
            time_string_bad = 'Tempo em ma postura : ' + str(round(bad_time, 1)) + 's'
            cv2.putText(image, time_string_bad, (10, h - 20), font, 0.9, red, 2)


        if bad_time > 180:
            sendWarning()

        boaPostura.append(good_time)
        maPostura.append(bad_time)


        video_output.write(image)
        stframe.image(image,channels="BGR",use_column_width=True)
        # ... (rest of your code for landmarks)
    else:
        print('*frame perdido')
        # Handle the case where landmarks are not detected in this frame
        continue  # Skip this frame and move on to the next one



alinhamento = (aliframe/(aliframe+desaliframe))*100
print(f"Porcentagem de alinhamento {alinhamento:.2f}")
df = pd.DataFrame({'Boa_Postura':boaPostura , 'Ma_postura': maPostura})
df.to_csv('./analisealpargatas.csv', index=False)
print('Finished.')
#print(f"total boa postura: {sum(boaPostura)}")
#print(f"total ma postura {sum(maPostura)}")
#print(f"media boa postura {max(boaPostura)}")
#print(f"media ma postura {max(maPostura)}")

cap.release()
video_output.release()
     # Atualize o gráfico de barras
boaPostura.append(good_frames)
maPostura.append(bad_frames)
num_bars = len(boaPostura)
bar_locations = range(num_bars)
plt.bar(bar_locations, boaPostura, width=0.5, color='blue', label='Boa postura')
plt.bar(bar_locations, maPostura, width=0.5, color='red', label='Má postura', bottom=boaPostura)
plt.title('Tempo em boa postura vs má postura')
chart.pyplot()




       

# Salve os dados em um arquivo CSV
df = pd.DataFrame({'Boa_Postura': boaPostura, 'Ma_Postura': maPostura})
st.sidebar.write('Data saved to CSV:')
st.sidebar.dataframe(df)

# Feche o vídeo e libere os recursos
cap.release()
video_output.release()
