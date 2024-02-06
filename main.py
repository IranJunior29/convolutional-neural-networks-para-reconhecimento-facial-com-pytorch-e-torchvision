# Imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torchvision.transforms.transforms import ToTensor
from tqdm import tqdm
from modelagem import ModeloCNN
from funcoes import multiclass_accuracy


# Função para o loop de treino
def treina_modelo(model, dataloader, optimizer, current_epo):

    # Coloca o modelo em modo de treino
    model.train()

    # Variáveis de controle
    total_loss = 0.0
    total_acc = 0.0

    # Mostra a barra de progressão durante o treino do modelo
    tk = tqdm(dataloader, desc="epoch" + "[treino]" + str(current_epo + 1) + "/" + str(epochs))

    # Loop
    for t, data in enumerate(tk):
        # Extrai lote de imagens e labels
        images, labels = data

        # Envia imagens e labels para a memória do device
        images, labels = images.to(device), labels.to(device)

        # Zera os gradientes
        optimizer.zero_grad()

        # Faz a previsão e retorna os logits
        logits, loss = model(images, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Calcula o erro
        total_loss += loss.item()

        # Calcula a acurácia
        total_acc += multiclass_accuracy(logits, labels)

        # Atualiza a barra de progressão
        tk.set_postfix({'loss': '%6f' % float(total_loss / (t + 1)), 'acc': '%6f' % float(total_acc / (t + 1))})

    return total_loss / len(dataloader), total_acc / len(dataloader)


# Função para o loop de validação
def avalia_modelo(model, dataloader, current_epo):
    # Coloca o modelo em modo de avaliação
    model.eval()

    # Inicializa as variáveis de controle
    total_loss = 0.0
    total_acc = 0.0

    # Barra de progressão
    tk = tqdm(dataloader, desc="epoch" + "[valid]" + str(current_epo + 1) + "/" + str(epochs))

    # Loop
    for t, data in enumerate(tk):
        # Extrai um lote de imagens e labels
        images, labels = data

        # Envia imagens e labels para a memória do device
        images, labels = images.to(device), labels.to(device)

        # Faz a previsão e retorna os logits
        logits, loss = model(images, labels)

        # Calcula erro e acurácia
        total_loss += loss.item()
        total_acc += multiclass_accuracy(logits, labels)

        # Atualiza a barra de progressão
        tk.set_postfix({'loss': '%6f' % float(total_loss / (t + 1)), 'acc': '%6f' % float(total_acc / (t + 1))})

    return total_loss / len(dataloader), total_acc / len(dataloader)

if __name__ == '__main__':
    # Verificando o dispositivo
    processing_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = processing_device
    print(device)

    ''' Organizando as Imagens em Disco '''

    # Dados tirado do ' https://www.kaggle.com/datasets/samaneheslamifar/facial-emotion-expressions?resource=download '
    # Pastas com as imagens
    train_img_folder_path = 'dados/train/'
    valid_img_folder_path = 'dados/validation/'

    ''' Dataset Augmentation '''

    # Dataset Augmentation em treino
    augmentation_treino = T.Compose([T.RandomHorizontalFlip(p=0.5),
                                     T.RandomRotation(degrees=(-20, +20)),
                                     T.ToTensor()])

    # Dataset Augmentation em validação
    augmentation_valid = T.Compose([ToTensor()])

    ''' Prepara os Datasets '''

    # Prepara os dados de treino
    dados_treino = ImageFolder(train_img_folder_path, transform=augmentation_treino)

    # Prepara os dados de validação
    dados_valid = ImageFolder(valid_img_folder_path, transform=augmentation_valid)

    print(f"Total de imagens no dataset de treino: {len(dados_treino)}")
    print(f"Total de imagens no dataset de validação: {len(dados_valid)}")

    # Índices das classes
    print(dados_treino.class_to_idx)

    ''' Preparando os DataLoaders '''

    # Hiperparâmetro
    BATCH_SIZE = 32

    # Criando os data loaders
    dl_treino = DataLoader(dados_treino, batch_size=BATCH_SIZE, shuffle=True)
    dl_valid = DataLoader(dados_valid, batch_size=BATCH_SIZE, shuffle=False)

    # Número de batches
    print(f"Número de Batches (Lotes) no DataLoader de Treino: {len(dl_treino)}")
    print(f"Número de Batches (Lotes) no DataLoader de Validação: {len(dl_valid)}")

    # Extraindo 1 batch do DataLoader de treino
    for images, labels in dl_treino:
        break;

    print(f"Shape de 1 Batch de Imagens: {images.shape}")
    print(f"Shape de 1 Batch de Labels das Imagens do Batch: {labels.shape}")

    ''' Modelagem '''

    # Cria uma instância da classe, um objeto
    modelo = ModeloCNN()

    # Envia o modelo para o device
    modelo.to(device)

    ''' Treinamento e Avaliação '''

    # Hiperparâmetros
    LR = 0.001
    epochs = 8

    # Otimizador
    optimizer = torch.optim.Adam(modelo.parameters(), lr=LR)

    # Inicializa a variável de controle
    best_valid_loss = np.Inf

    # Iniciando o treinamento
    for i in range(epochs):

        # Loop de treino
        train_loss, train_acc = treina_modelo(modelo, dl_treino, optimizer, i)

        # Loop de validação
        valid_loss, valid_acc = avalia_modelo(modelo, dl_valid, i)

        # Salva o melhor modelo
        if valid_loss < best_valid_loss:
            # Salva o modelo
            torch.save(modelo, 'modelo/melhor_modelo.pt')

            print("Salvando o melhor modelo em disco...")

            best_valid_loss = valid_loss














