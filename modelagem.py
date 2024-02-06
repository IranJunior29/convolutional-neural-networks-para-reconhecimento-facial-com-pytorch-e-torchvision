# Imports
import timm
from torch import nn

''' Modelagem '''

# Classe com arquitetura do modelo
class ModeloCNN(nn.Module):

    # Método construtor
    def __init__(self):
        # Inicializa a classe mãe (nn.Module)
        super(ModeloCNN, self).__init__()

        # Carrega o modelo pré-treinado
        self.eff_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)

    # Método forward
    # Observe que labels = None, pois estamos fazendo exatamente as previsões dos labels
    def forward(self, images, labels=None):
        # Extrai as previsões (logits)
        logits = self.eff_net(images)

        # Calcula o erro do modelo
        if labels != None:
            loss = nn.CrossEntropyLoss()(logits, labels)

            return logits, loss

        return logits
