# Imports
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.transforms.transforms import ToTensor

if __name__ == '__main__':

    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = processing_device
    print(device)

    ''' Deploy Para Uso do Modelo Treinado com Novas Imagens '''

    augmentation_valid = T.Compose([ToTensor()])
    valid_img_folder_path = 'dados/validation/'
    dados_valid = ImageFolder(valid_img_folder_path, transform=augmentation_valid)
    BATCH_SIZE = 32
    dl_valid = DataLoader(dados_valid, batch_size=BATCH_SIZE, shuffle=False)

    # Carrega o modelo treinado
    modelo_final = torch.load('modelo/melhor_modelo.pt')

    # Carrega um batch de imagens
    for image, label in dl_valid:
        break;

    # Coloca imagem e label no device
    image, label = image.to(device), label.to(device)

    # Previsão com o modelo treinado (a previsão é para um batch de dados)
    with torch.no_grad():
        previsao = modelo_final(image)

    # Captura uma imagem do lote de previsões
    uma_imagem = previsao[6]

    # São 7 previsões de classe para cada imagem. Aqui temos os logits
    print(uma_imagem)

    # Converte os logits em probabilidades
    probs = torch.nn.functional.softmax(uma_imagem, dim=-1)

    print(probs)

    # Extrai a maior probabilidade das previsões de classe da imagem
    _, classe_predita = torch.max(probs, 0)

    print(f"A classe predita para a imagem é: {classe_predita}")

    # Índices das classes
    print(dados_valid.class_to_idx)

    # Plot da imagem e do label original
    image, label = dados_valid[6]
    plt.imshow(image.permute(1, 2, 0))
    plt.title(label)
