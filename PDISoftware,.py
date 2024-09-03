import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Inicializar o subtrator de fundo
fgbg = cv2.createBackgroundSubtractorMOG2()

# Variáveis globais para contadores
contadores = {
    "Adulto": 0,
    "Criança": 0,
    "Animal": 0,
    "Desconhecido": 0
}

# Conjunto para rastrear contornos únicos
contornos_exibidos = set()

def classify_contour(contour):
    # Calcula a área do contorno
    area = cv2.contourArea(contour)
    
    # Obtém o retângulo delimitador
    x, y, w, h = cv2.boundingRect(contour)
    
    # Calcula a proporção largura/altura
    aspect_ratio = float(w) / h

    # Define limites de área para classificação
    area_adulto_max = 1500  # Área máxima para adultos
    area_crianca_max = 800  # Área máxima para crianças (ajustado)
    area_animal_max = 1200  # Área máxima para animais (ajustado)

    if area > 5000:  # Limite de área para grandes objetos
        return "Desconhecido"

    if aspect_ratio > 1.5:  # Largura maior que altura (possível animal ou humano deitado)
        if area <= area_animal_max:  # Limite de área para animal
            return "Animal"
        else:
            return "Desconhecido"  # Pode ser ruído ou objeto grande
    else:  # Altura maior que largura (possível humano em pé ou criança)
        if area <= area_crianca_max:  # Área menor pode ser uma criança
            return "Criança"
        elif area <= area_adulto_max:  # Área maior pode ser um adulto
            return "Adulto"
        else:
            return "Desconhecido"

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

        # Limpar contadores a cada frame
        for key in contadores:
            contadores[key] = 0

        # Aplicar filtro para escala de cinza
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar filtro bilateral para suavizar a imagem
        bilateral_frame = cv2.bilateralFilter(gray_frame, 9, 75, 75)

        # Aplicar o algoritmo de detecção de bordas Canny
        edges = cv2.Canny(bilateral_frame, 50, 150)

        # Aplicar limiar binário
        _, binary_mask = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

        # Aplicar operações morfológicas para limpar a imagem
        kernel = np.ones((3, 3), np.uint8)  # Kernel menor para limpeza mais adequada
        dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)  # Dilatação para reforçar contornos
        cleaned_mask = cv2.erode(dilated_mask, kernel, iterations=1)   # Erosão para limpar pequenos ruídos

        # Encontrar contornos
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Exibir as diferentes fases do processamento
        cv2.imshow('Original', frame)
        cv2.imshow('Escala de Cinza', gray_frame)
        cv2.imshow('Filtro Bilateral', bilateral_frame)
        cv2.imshow('Bordas Canny', edges)
        cv2.imshow('Máscara Binária', binary_mask)
        cv2.imshow('Máscara Limpa', cleaned_mask)
        
        # Exibir a imagem com contornos e classificações
        frame_with_contours = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filtrar pequenos contornos
                # Classificar o contorno
                label = classify_contour(contour)
                
                # Obter a caixa delimitadora do contorno
                x, y, w, h = cv2.boundingRect(contour)
                contour_id = (x, y, w, h)
                
                # Verificar se o contorno já foi exibido
                if contour_id not in contornos_exibidos:
                    # Incrementar o contador do tipo identificado
                    contadores[label] += 1
                    
                    # Adicionar o contorno ao conjunto de contornos exibidos
                    contornos_exibidos.add(contour_id)
                    
                    # Desenhar o retângulo delimitador e a etiqueta
                    cv2.rectangle(frame_with_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame_with_contours, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Atualizar a interface com as contagens do frame atual
        counter_text = (f"Adulto: {contadores['Adulto']}\n"
                        f"Criança: {contadores['Criança']}\n"
                        f"Animal: {contadores['Animal']}\n"
                        f"Desconhecido: {contadores['Desconhecido']}")

        # Desenhar o texto dos contadores sobre o frame
        y0, dy = 30, 30
        for i, line in enumerate(counter_text.split('\n')):
            y = y0 + i * dy
            cv2.putText(frame_with_contours, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Detecção de Pessoas e Animais', frame_with_contours)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Configuração da Janela Principal
root = tk.Tk()
root.title("Detecção de Pessoas e Animais")
root.geometry("300x150")

# Definindo a largura dos botões
button_width = 20

# Botões da Interface
btn_start = tk.Button(root, text="Iniciar Processamento", command=start_processing, width=button_width)
btn_start.pack(pady=20)

btn_exit = tk.Button(root, text="Sair", command=root.quit, width=button_width)
btn_exit.pack(pady=10)

# Inicia o Loop Principal da Interface
root.mainloop()
