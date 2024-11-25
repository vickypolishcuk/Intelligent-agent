import numpy as np

# Ініціалізація матриці R
R = np.array([
    [-1, 0, -1, -1, -1, -1, -1, -1, -1, 100],
    [0, -1, 0, -1, -1, -1, -1, -1, -1, -1],
    [-1, 0, -1, 0, -1, -1, -1, -1, -1, -1],
    [-1, -1, 0, -1, 0, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 0, -1, -1, -1, -1],
    [-1, -1, -1, -1, 0, -1, 0, -1, -1, -1],
    [-1, -1, -1, -1, -1, 0, -1, 0, -1, -1],
    [-1, -1, -1, -1, -1, -1, 0, -1, 0, 100],
    [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1],
    [0, -1, -1, -1, -1, -1, -1, 0, -1, 100],
])

# Ініціалізація матриці Q нулями
Q = np.zeros_like(R)

# Параметр швидкості навчання
gamma = 0.8

# Функція для отримання можливих дій для даного стану
def available_actions(state):
    return np.where(R[state] >= 0)[0]

# Функція для вибору наступної дії
def choose_action(available_actions):
    return np.random.choice(available_actions)

# Функція для оновлення матриці Q
def update_Q(state, action, gamma):
    max_index = np.argmax(Q[action])
    Q[state, action] = R[state, action] + gamma * Q[action, max_index]

# Навчання агента
def train(final_state=9, generations=30):
    for _ in range(generations):  # умовою виходу є досягнення кількості епох навчання
        current_state = np.random.randint(0, final_state+1)  # випадковий початковий стан
        while current_state != final_state:  # поки агент не досягне цільового стану
            action = choose_action(available_actions(current_state))
            update_Q(current_state, action, gamma)
            current_state = action
            if current_state == final_state:
                action = choose_action(available_actions(current_state))
                update_Q(current_state, action, gamma)
                current_state = action

# Тестування агента
def test(current_state=0, final_state=9):
    print("Початковий стан:", current_state)
    while current_state != final_state:  # поки агент не досягне цільового стану
        action = np.argmax(Q[current_state])
        current_state = action
        print("Вибрана дія:", action, "Новий стан:", current_state)
    print("Агент досяг виходу.")

# Тренування агента
train()

# Виведення матриці вагів
print("Отримана матриця Q:")
print(Q)

# Тестування агента по всім можливим станам
for i in range(0, 10):
    test(i)

