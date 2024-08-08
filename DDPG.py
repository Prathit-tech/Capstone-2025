import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam

# Hyperparameters
STATE_DIM = 12  # Ego vehicle state + pedestrian/driver intentions + gap info
ACTION_DIM = 3  # Accelerate, decelerate, maintain
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.001
LR_ACTOR = 0.0001
LR_CRITIC = 0.001

class DDPG:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Actor Network
        self.actor = self.build_actor_model()
        self.target_actor = self.build_actor_model()

        # Critic Network
        self.critic = self.build_critic_model()
        self.target_critic = self.build_critic_model()

        # Initialize target model parameters to match primary models
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.replay_buffer = []

    def build_actor_model(self):
        inputs = Input(shape=(self.state_dim,))
        x = Dense(256, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='tanh')(x)
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(lr=LR_ACTOR), loss='mse')
        return model

    def build_critic_model(self):
        state_input = Input(shape=(self.state_dim,))
        action_input = Input(shape=(self.action_dim,))
        x1 = Dense(256, activation='relu')(state_input)
        x2 = Dense(256, activation='relu')(action_input)
        x = Concatenate()([x1, x2])
        x = Dense(128, activation='relu')(x)
        outputs = Dense(1, activation=None)(x)
        model = Model(inputs=[state_input, action_input], outputs=outputs)
        model.compile(optimizer=Adam(lr=LR_CRITIC), loss='mse')
        return model

    def act(self, state):
        return self.actor.predict(state)[0]

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay(self):
        minibatch = np.random.choice(len(self.replay_buffer), BATCH_SIZE)
        states = np.zeros((BATCH_SIZE, self.state_dim))
        actions = np.zeros((BATCH_SIZE, self.action_dim))
        rewards = np.zeros((BATCH_SIZE, 1))
        next_states = np.zeros((BATCH_SIZE, self.state_dim))
        dones = np.zeros((BATCH_SIZE, 1))

        for i, idx in enumerate(minibatch):
            state, action, reward, next_state, done = self.replay_buffer[idx]
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done

        target_actions = self.target_actor.predict(next_states)
        target_q_values = self.target_critic.predict([next_states, target_actions])

        y_i = rewards + GAMMA * target_q_values * (1 - dones)

        self.critic.fit([states, actions], y_i, epochs=1, verbose=0)

        actions_for_gradients = self.actor.predict(states)
        grads = self.critic.gradient([states, actions_for_gradients])
        self.actor.train_on_batch(states, -grads)

        self.update_target_models()

    def update_target_models(self):
        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()
        self.target_actor.set_weights([
            (1 - TAU) * taw + TAU * aw for taw, aw in zip(self.target_actor.get_weights(), actor_weights)
        ])
        self.target_critic.set_weights([
            (1 - TAU) * taw + TAU * aw for taw, aw in zip(self.target_critic.get_weights(), critic_weights)
        ])
