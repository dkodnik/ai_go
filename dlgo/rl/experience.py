import numpy as np

class ExperienceBuffer:
    """Минимальный контейнер для хранения набора данных опыта"""
    def __init__(self, states, actions, rewards):
        self.states = states
        self.actions = actions
        self.rewards = rewards

    def serialize(self, h5file):
        """Сохранение буфера опыта на диск"""
        h5file.create_group('experience')
        h5file['experience'].create_dataset(
            'states', data=self.states)
        h5file['experience'].create_dataset(
            'actions', data=self.actions)
        h5file['experience'].create_dataset(
            'rewards', data=self.rewards)

class ExperienceCollector:
    """Объект для отслеживания решений, принятых в ходе одного эпизода"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.current_episode_states = []
        self.current_episode_actions = []

    def begin_episode(self):
        """Вызывается драйвером игры бота с самим собой для обозначения начала отдельной партии"""
        self.current_episode_states = []
        self.current_episode_actions = []

    def record_decision(self, state, action):
        """Вызывается агентом для обозначения одного выбранного им действия"""
        #сохранение одного решения в текущем эпизоде; за кодирование состояния и действия
        #отвечает агент
        self.current_episode_states.append(state)
        self.current_episode_actions.append(action)

    def complete_episode(self, reward):
        """Вызывается драйвером игры бота с самим собой для обозначения окончания отдельной партии"""
        num_states = len(self.current_episode_states)
        self.states += self.current_episode_states
        self.actions += self.current_episode_actions
        #распределение итоговой награды среди всех действий, совершенных в ходе партии
        self.rewards += [reward for _ in range(num_states)]

        self.current_episode_states = []
        self.current_episode_actions = []

    def to_buffer(self):
        """Вызывается по завершению партии. Упаковываются все данные, собранные
        объектом ExperienceCollector и возвращает ExperienceBuffer.
        """
        #накапливаются списки Python; при этом они преобразуются в массив NumPy
        return ExperienceBuffer(
            states=np.array(self.states),
            actions=np.array(self.actions),
            rewards=np.array(self.rewards))

def combine_experience(collectors):
    combined_states = np.concatenate([np.array(c.states) for c in collectors])
    combined_actions = np.concatenate([np.array(c.actions) for c in collectors])
    combined_rewards = np.concatenate([np.array(c.rewards) for c in collectors])
    #combined_advantages = np.concatenate([np.array(c.advantages) for c in collectors])

    return ExperienceBuffer(
        combined_states,
        combined_actions,
        combined_rewards,
        #combined_advantages,
    )

def load_experience(h5file):
    """Восстановление ExperienceBuffer из файла HDF5"""
    return ExperienceBuffer(
        states=np.array(h5file['experience']['states']),
        actions=np.array(h5file['experience']['actions']),
        rewards=np.array(h5file['experience']['rewards']),
    )