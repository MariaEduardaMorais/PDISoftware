import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Inicializar o subtrator de fundo
fgbg = cv2.createBackgroundSubtractorMOG2()

def classify_contour(contour):
    # Calcula a área do contorno
    area = cv2.contourArea(contour)
    
    # Obtém o retângulo delimitador
    x, y, w, h = cv2.boundingRect(contour)

    # Classifica com base na proporção largura/altura
    aspect_ratio = float(w) / h
    
    if aspect_ratio > 1.5:  # Largura maior que altura (pode ser um animal ou humano deitado)
        if area > 5000:  # Ajuste o limite de área conforme necessário
            return "Animal"
        else:
            return "Humano deitado"
    else:  # Altura maior que largura (pode ser um humano em pé)
        return "Humano em pé"

def start_processing():
    video_path = filedialog.askopenfilename(title="Select a Video File",
                                            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*")))
    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Aplicar o filtro GaussianBlur para suavizar a imagem
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Aplicar a subtração de fundo
        fgmask = fgbg.apply(blurred_frame)
        
        # Aplicar operações morfológicas para limpar a imagem
        kernel = np.ones((7, 7), np.uint8)  # Kernel maior para limpeza mais robusta
        cleaned_mask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Exibir as diferentes fases do processamento
        cv2.imshow('Original', frame)
        cv2.imshow('Subtração de Fundo', fgmask)
        cv2.imshow('Máscara Limpa', cleaned_mask)
        
        # Exibir a imagem com contornos e classificações
        frame_with_contours = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filtrar pequenos contornos
                # Classificar o contorno
                label = classify_contour(contour)
                
                # Desenhar o retângulo delimitador e a etiqueta
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_with_contours, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Detecção de Pessoas e Animais', frame_with_contours)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def show_info():
    messagebox.showinfo("About", "Software de Detecção e Contagem de Pessoas e Animais sem Machine Learning. Desenvolvido com OpenCV e Tkinter.")

# Configuração da Janela Principal
root = tk.Tk()
root.title("Detecção de Pessoas e Animais")
root.geometry("300x200")

# Botões da Interface
btn_start = tk.Button(root, text="Iniciar Processamento", command=start_processing)
btn_start.pack(pady=20)

btn_info = tk.Button(root, text="Sobre", command=show_info)
btn_info.pack(pady=10)

btn_exit = tk.Button(root, text="Sair", command=root.quit)
btn_exit.pack(pady=10)

# Inicia o Loop Principal da Interface
root.mainloop()
