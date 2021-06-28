import numpy as np


drinks = 1.0
rain = 1.0
friend = 1.0

def activation_function(x):#старая активационная функция без сигмоиды
    if x >= 0.5:
        return 1
    else:
        return 0

def predict(drinks, rain, friend):
    inputs = np.array([drinks, rain, friend])#создание матрицы из входных даных
    weights_inputs_to_hiden_1 = [0.25, 0.25, 0]#весы для первого скрытого нейрона
    weights_inputs_to_hiden_2 = [0.5, -0.4, 0.9]#весы для второго скрытого нейрона 
    weights_inputs_to_hiden = np.array([weights_inputs_to_hiden_1, weights_inputs_to_hiden_2])#создание матрицы из весов

    weights_hiden_to_output = np.array([-1, 1])#матрица из весов на выходы 

    hiden_input = np.dot(weights_inputs_to_hiden, inputs)#перемножение двух матриц
    print('hiden output: ' + str(hiden_input))

    hiden_output = np.array([activation_function(x) for x in hiden_input])#проход через активационную функцию
    print('hiden output: ' + str(hiden_output))

    output = np.dot(weights_hiden_to_output, hiden_output)#перемножение матриц 
    print('output: ' + str(output))
    return activation_function(output) == 1

print('result: ' + str(predict(drinks, rain, friend)))

train_values = (
    (1, 0, 0, True),
    (0, 1, 0, False),
    (0, 0, 1, True),
    (1, 1, 0, False),
    (0, 1, 1, True),
    (1, 1, 1, False)
)

for value in train_values:
    print('\ndrinks: {}; rain: {}; friend: {};\noutput: {}\nexpected: {}\n----------'.format(value[0], value[1], value[2], predict(value[0], value[1], value[2]), value[3]))
