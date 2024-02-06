# Imports
import torch


''' Funções Para o Loop de Treino e Avaliação '''


# Função para calcular a acurácia da classificação multiclasse
def multiclass_accuracy(y_pred, y_true):

    # top classes
    top_p, top_class = y_pred.topk(1, dim=1)

    # classes iguais
    equals = top_class == y_true.view(*top_class.shape)

    # Retorna a média
    return torch.mean(equals.type(torch.FloatTensor))
