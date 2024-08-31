import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# Inicializar o subtrator de fundo
fgbg = cv2.createBackgroundSubtractorMOG2()

# Variáveis globais para contadores
contadores = {
    "Humano em pé": 0,
    "Humano deitado": 0,
    "Criança": 0,
    "Animal": 0,
    "Desconhecido": 0
}

def classify_contour(contour):
    # Calcula a área do contorno
    area = cv2.contourArea(contour)
    
    # Obtém o retângulo delimitador
    x, y, w, h = cv2.boundingRect(contour)

    # Classifica com base na proporção largura/altura e na área
    aspect_ratio = float(w) / h
    
    if aspect_ratio > 1.5:  # Largura maior que altura (pode ser um animal ou humano deitado)
        if area > 5000:  # Ajuste o limite de área conforme necessário
            return "Animal"
        else:
            return "Humano deitado"
    else:  # Altura maior que largura (pode ser um humano em pé)
        if area > 1500:  # Definimos um limite de área para adultos e crianças
            if h > 150:  # Altura mínima para ser considerado um adulto
                return "Humano em pé"
            else:
                return "Criança"
        else:
            return "Desconhecido"  # Pode ser ruído ou um objeto muito pequeno

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

        # Reiniciar contadores a cada frame
        for key in contadores:
            contadores[key] = 0

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
                
                # Incrementar o contador do tipo identificado
                contadores[label] += 1
                
                # Desenhar o retângulo delimitador e a etiqueta
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame_with_contours, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Detecção de Pessoas e Animais', frame_with_contours)

        # Atualizar a interface com as contagens do frame atual
        update_counters()

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def update_counters():
    # Atualiza o texto dos contadores na interface
    counter_text.set(f"Humano em pé: {contadores['Humano em pé']}\n"
                     f"Humano deitado: {contadores['Humano deitado']}\n"
                     f"Criança: {contadores['Criança']}\n"
                     f"Animal: {contadores['Animal']}\n"
                     f"Desconhecido: {contadores['Desconhecido']}")

def show_info():
    messagebox.showinfo("About", "Software de Detecção e Contagem de Pessoas e Animais sem Machine Learning. Desenvolvido com OpenCV e Tkinter.")

def open_counter_window():
    counter_window = tk.Toplevel(root)
    counter_window.title("Contadores")
    counter_window.geometry("300x200")

    # Rótulo para exibir contadores na janela separada
    lbl_counters = tk.Label(counter_window, textvariable=counter_text)
    lbl_counters.pack(pady=20)

# Configuração da Janela Principal
root = tk.Tk()
root.title("Detecção de Pessoas e Animais")
root.geometry("300x200")

# Definindo a largura dos botões
button_width = 20

# Texto para exibir contadores
counter_text = tk.StringVar()
counter_text.set("Humano em pé: 0\nHumano deitado: 0\nCriança: 0\nAnimal: 0\nDesconhecido: 0")

# Botões da Interface
btn_start = tk.Button(root, text="Iniciar Processamento", command=start_processing, width=button_width)
btn_start.pack(pady=10)

btn_counters = tk.Button(root, text="Exibir Contadores", command=open_counter_window, width=button_width)
btn_counters.pack(pady=10)

btn_info = tk.Button(root, text="Sobre", command=show_info, width=button_width)
btn_info.pack(pady=10)

btn_exit = tk.Button(root, text="Sair", command=root.quit, width=button_width)
btn_exit.pack(pady=10)

# Inicia o Loop Principal da Interface
root.mainloop()
