# 1. Definir as listas x e y
x = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# 2. Calcular as médias de x e y
media_x = sum(x) / len(x)
media_y = sum(y) / len(y)

# 3. Inicializar as variáveis para os somatórios
soma_num = 0  
soma_den = 0  

# 4. Utilizar um loop para calcular:
for i in range(len(x)):
    soma_num += (x[i] - media_x) * (y[i] - media_y)
    soma_den += (x[i] - media_x) ** 2

# 5. Calcular beta1 e beta0 usando as fórmulas dos mínimos quadrados
beta1 = soma_num / soma_den
beta0 = media_y - beta1 * media_x

# 6. Imprimir os resultados
print(f"Coeficiente beta0 (intercepto): {beta0}")
print(f"Coeficiente beta1 (inclinação): {beta1}") 