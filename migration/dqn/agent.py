from migration.dqn.brain import DQNMigrationBrain


class DQNMigrationAgent:
    def __init__(self, dim_mig_nn_input, decaying_cnt):
        '''태스크의 상태 및 행동의 가짓수를 설정'''
        self.brain = DQNMigrationBrain(dim_mig_nn_input)  # 에이전트의 행동을 결정할 두뇌 역할 객체를 생성
        self.decaying_cnt = decaying_cnt

    def update_q_function(self):
        '''Q함수를 수정'''
        self.brain.replay()

    def get_action(self, state):
        '''행동을 결정'''
        action = self.brain.decide_action(state, self.decaying_cnt)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memory 객체에 state, action, state_next, reward 내용을 저장'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        '''Target Q-Network을 Main Q-Network와 맞춤'''
        self.brain.update_target_q_network()
