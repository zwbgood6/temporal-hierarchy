import cv2
import numpy as np


class BouncingBalls(object):
    """
    Bouncing ball environment: generates binary images of balls bouncing in a fixed frame.
    Action space: [right, left, up, down]
    """

    # Valid action space is [right, left, up, down, idle].
    num_actions = 5
    max_reward = 1.0
    max_num_steps = 10000  # eventually start the new episode if the agent can't reach the target

    def __init__(self,
                 resolution=64,
                 num_balls=2,
                 ball_size=0.1,
                 agent_size=(1/20),
                 ball_speed=1.0,
                 agent_speed=0.25,
                 stochastic_angle=False,
                 random_angle=False,
                 stochastic_speed=False,
                 stochastic_bounce=False,
                 rand_start_counter=False,
                 variance_degrees=5,  # angle variance in degree
                 bounce_vertically=False,
                 segment_length_limits=None,
                 render=True,
                 hide_target=True,
                 hide_agent=True,
                 seed=None):
        self._num_steps = 0
        self._resolution = resolution
        self._num_balls = num_balls
        self._stochastic_angle = stochastic_angle
        self._random_angle = random_angle
        self._stochastic_speed = stochastic_speed
        self._stochastic_bounce = stochastic_bounce
        self._rand_start_counter = rand_start_counter
        self._variance = variance_degrees * np.pi / 180.0
        self._bounce_vertically = bounce_vertically
        self._segment_length_limits = segment_length_limits
        self._render_output = render
        self._hide_target = hide_target
        self._hide_agent = hide_agent
        self._ball_r = int(self._resolution * ball_size)    # heuristic for a reasonable ball size
        self._ball_base_v = ball_speed * self._ball_r
        self._agent_v = agent_speed * self._ball_r
        self._agent_size = agent_size
        self._margin = 1 * self._ball_r     # margin on the sides of the frame

        if self._stochastic_angle and self._bounce_vertically or \
                self._random_angle and self._bounce_vertically:
            raise NotImplementedError("Stochastic/Random angle and vertical bounce cannot be enabled at the same time.")

        # initialize random number generator
        self.seed(seed)

        # initialize state
        self._state = dict()
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        self._num_steps = 0

        self._state['goal'] = self.gen_rand_pos()
        self._state['agent_pos'] = self.gen_rand_pos()      # format: [x,y]
        self._state['ball_pos'] = [self.gen_rand_pos() for _ in range(self._num_balls)]

        if self._bounce_vertically:
            self._state['ball_angle'] = [np.pi/2.0 if np.random.rand() < 0.5 else -np.pi/2.0 for _ in range(self._num_balls)]
        else:
            self._state['ball_angle'] = [self.gen_rand_angle() for _ in range(self._num_balls)]

        if self._stochastic_speed:
            self._state['ball_speed'] = [self.gen_rand_speed() for _ in range(self._num_balls)]
        else:
            self._state['ball_speed'] = [self._ball_base_v for _ in range(self._num_balls)]

        if self._stochastic_bounce:
            self._state['bounce_counter'] = [self.gen_rand_bounce_counter(pos, speed, angle, i) for
                                             i, (pos, speed, angle) in enumerate(zip(self._state['ball_pos'],
                                                                        self._state['ball_speed'],
                                                                        self._state['ball_angle']))]
            if self._rand_start_counter:
                # modify bounce_counter to have random length of first segment (makes learning of fixed pattern harder)
                self._state['bounce_counter'] = [np.random.randint(1, bc+1) for bc in self._state['bounce_counter']]
        else:
            self._state['bounce_counter'] = [None for _ in range(self._num_balls)]

        if self._render_output:
            obs = self.render(mode='rgb_array')
        else:
            obs = self.normalize_vec(self.state2vec(self._state))
        return obs

    def gen_rand_pos(self):
        return np.round(np.random.uniform(self._margin, self._resolution - self._margin, size=2))

    def gen_rand_angle(self):
        return (np.random.rand()-0.5) * 2 * np.pi

    def gen_rand_speed(self):
        return (np.random.rand() + 1) * self._ball_base_v   # range 1..2 * base_v

    def comp_free_length(self, pos, angle):
        """Computes the length of free space until next boundary is hit."""
        if angle > 0:   # upper hemisphere
            dy = pos[1] - self._margin
        else:   # lower hemisphere
            dy = (self._resolution - self._margin) - pos[1]
        if np.abs(angle) > np.pi/2:     # left hemisphere
            dx = pos[0] - self._margin
        else:       # right hemisphere
            dx = (self._resolution - self._margin) - pos[0]

        ddx = np.abs(dy / np.tan(angle))
        ddy = np.abs(np.tan(angle) * dx)

        if ddx > dx:
            free_length = np.sqrt(dx**2 + ddy**2)
        else:
            free_length = np.sqrt(ddx**2 + dy**2)
        return free_length

    def gen_rand_bounce_counter(self, ball_pos, ball_speed, ball_angle, ball_idx):
        dd = self.comp_free_length(ball_pos, ball_angle)
        max_counter = np.floor(dd / ball_speed)     # maximal number of steps with this speed without hitting a border
        if self._random_angle:
            fallback_cnt = 0
            while max_counter < self._segment_length_limits[0]:
                ball_angle = self.gen_rand_angle()
                self._state['ball_angle'][ball_idx] = ball_angle
                dd = self.comp_free_length(ball_pos, ball_angle)
                max_counter = np.floor(dd / ball_speed)
                fallback_cnt += 1
                if fallback_cnt > 1000:     # in case while loop gets stuck for some reason
                    raise ValueError('Could not find a valid new bouncing angle!')
        if max_counter == 0 or max_counter == 1:
            return 1     # for cases when ball spawns too close to border
        if self._segment_length_limits is not None:
            if max_counter <= self._segment_length_limits[0]:
                return max_counter  # return max number of steps if fewer than desired
            if max_counter > self._segment_length_limits[1]:
                max_counter = self._segment_length_limits[1]
        return np.random.randint(self._segment_length_limits[0], max_counter+1)   # top limit is +1 from highest sampled

    def clip(self, pos):
        return np.clip(pos, self._margin, self._resolution - self._margin)

    def step(self, scalar_action):
        # small penalty for every move, standing still gets slightly less penalty
        action_penalty = -0.002 * self.max_reward
        if scalar_action == self.num_actions - 1:
            action_penalty /= 2

        action = np.zeros((self.num_actions,))
        action[scalar_action] = 1

        assert action.shape == (self.num_actions,), "Valid action space is [right, left, up, down, idle]."
        assert not (action < 0).any() and np.sum(action) == 1, \
            "Action input can only contain a single element that is set to 1, all others are 0."

        self._num_steps += 1

        # update agent position
        self._state['agent_pos'][0] = self.clip(self._state['agent_pos'][0] + self._agent_v * (action[0] - action[1]))
        self._state['agent_pos'][1] = self.clip(self._state['agent_pos'][1] + self._agent_v * (action[3] - action[2]))

        # update ball positions + angle
        bounce_happened = False
        for i, (ball_pos, ball_angle, ball_speed, bounce_counter) in \
            enumerate(zip(self._state['ball_pos'], self._state['ball_angle'],
                          self._state['ball_speed'], self._state['bounce_counter'])):
            updated_pos_x = ball_pos[0] + np.cos(ball_angle) * ball_speed
            updated_pos_y = ball_pos[1] - np.sin(ball_angle) * ball_speed
            if self._stochastic_bounce:
                self._state['bounce_counter'][i] -= 1
            if updated_pos_x < self._margin or updated_pos_x > self._resolution - self._margin:
                # bounce on left or right border -> invert velocity in x direction
                bounce_happened = True
                self._state['ball_angle'][i] = np.arctan2(np.sin(ball_angle), -np.cos(ball_angle))
                if self._stochastic_angle:
                    self._state['ball_angle'][i] += np.random.normal(loc=0.0, scale=self._variance)
                if self._random_angle:
                    self._state['ball_angle'][i] = self.gen_rand_angle()
                if self._stochastic_speed and updated_pos_x < self._margin:     # only change speed on left bounce
                    self._state['ball_speed'][i] = self.gen_rand_speed()
            if self._stochastic_bounce and self._state['bounce_counter'][i] == 0 or \
                    updated_pos_y < self._margin or updated_pos_y > self._resolution - self._margin:
                # bounce on bottom or top border -> invert velocity in y direction
                bounce_happened = True
                self._state['ball_angle'][i] = np.arctan2(-np.sin(ball_angle), np.cos(ball_angle))
                if self._stochastic_angle:
                    self._state['ball_angle'][i] += np.random.normal(loc=0.0, scale=self._variance)
                if self._random_angle:
                    self._state['ball_angle'][i] = self.gen_rand_angle()
                if self._stochastic_speed and updated_pos_y < self._margin: # only change speed on top bounce
                    self._state['ball_speed'][i] = self.gen_rand_speed()
                if self._stochastic_bounce:
                    self._state['bounce_counter'][i] = self.gen_rand_bounce_counter([updated_pos_x, updated_pos_y],
                                                                                    self._state['ball_speed'][i],
                                                                                    self._state['ball_angle'][i], i)
            # update values
            self._state['ball_pos'][i][0] = self.clip(updated_pos_x)
            self._state['ball_pos'][i][1] = self.clip(updated_pos_y)

        # compute reward -> check for collisions with balls/target
        done = False
        reward = action_penalty
        for ball_pos in self._state['ball_pos']:
            if np.linalg.norm(ball_pos - self._state['agent_pos']) <= self._ball_r * 2:
                reward = -self.max_reward
                done = True
                break
        if np.linalg.norm(self._state['goal'] - self._state['agent_pos']) <= self._ball_r * 2:
            reward = self.max_reward
            done = True

        if not done and self._num_steps >= self.max_num_steps:
            done = True

        # compute/render output observation
        if self._render_output:
            obs = self.render(mode='rgb_array')
        else:
            obs = self.normalize_vec(self.state2vec(self._state))

        # just in case, clip reward
        reward = np.clip(reward, -self.max_reward, self.max_reward)

        return obs, [np.cos(ball_angle) * ball_speed,  - np.sin(ball_angle) * ball_speed], int(bounce_happened), {}

    def state2vec(self, state):
        """generates vector output: [ball_pos (num_balls x 2), agent_pos (2), goal_pos (num_goals x 2)]"""
        output_vector_size = self.output_size[0]
        output_vector = np.empty((output_vector_size,), dtype=self.output_type)
        current_idx = 0

        for ball_idx in range(self._num_balls):
            output_vector[current_idx:current_idx + 2] = state['ball_pos'][ball_idx]
            current_idx += 2

        if not self._hide_agent:
            output_vector[current_idx:current_idx + 2] = state['agent_pos']
            current_idx += 2

        if not self._hide_target:
            output_vector[current_idx:current_idx + 2] = state['goal_pos']
            current_idx += 2

        return output_vector

    def vec2state(self, vec):
        # converts a vector input to a state representation
        state = dict()
        current_idx = 0

        state['ball_pos'] = []
        for ball_idx in range(self._num_balls):
            state['ball_pos'].append(vec[current_idx:current_idx + 2])
            current_idx += 2

        if not self._hide_agent:
            state['agent_pos'] = vec[current_idx:current_idx + 2]
            current_idx += 2

        if not self._hide_target:
            state['goal_pos'] = vec[current_idx:current_idx + 2]
            current_idx += 2

        return state

    def render_state(self, state, render_res, scale, ball_r, agent_sz):
        img = np.zeros((render_res, render_res), dtype=float)

        # plot balls
        for ball_pos in state["ball_pos"]:
            disc_ball_pos = [int(ball_pos[0] * scale), int(ball_pos[1] * scale)]
            cv2.circle(img, tuple(disc_ball_pos), ball_r, 1.0, thickness=-1)

        # plot agent and goal
        if not self._hide_agent:
            agent_pos = state["agent_pos"]
            img = cv2.rectangle(
                img,
                tuple([int(agent_pos[0] * scale) - agent_sz,
                       int(agent_pos[1] * scale) - agent_sz]),
                tuple([int(agent_pos[0] * scale) + agent_sz,
                       int(agent_pos[1] * scale) + agent_sz]), 1.0, thickness=-1)

        if not self._hide_target:
            for goal in state["goal_pos"]:
                goal_pts = np.array(
                    [[int(goal[0] * scale - agent_sz), int(goal[1] * scale + agent_sz)],
                     [int(goal[0] * scale + agent_sz), int(goal[1] * scale + agent_sz)],
                     [int(goal[0] * scale), int(goal[1] * scale - agent_sz)]], np.int32)
                goal_pts = goal_pts.reshape((-1, 1, 2))
                cv2.polylines(img, [goal_pts], True, 1.0, thickness=min(3, scale))

        return np.expand_dims(img, axis=0)

    def render_state_vec(self, state_vector):
        # renders a scene image from a given state vector
        state_vector = self.unnormalize_vec(state_vector)
        state = self.vec2state(state_vector)
        img = self.render_state(
            state, render_res=self._resolution, scale=1.0, ball_r=self._ball_r, agent_sz=self._agent_size,
        )
        return img

    def render_state_vec_batch(self, state_vec_batch):
        # renders a batch of state vectors of size [batch_size, timesteps, vec_dim]
        batch_size, num_steps, vec_dim = state_vec_batch.shape
        output_imgs = np.empty([batch_size, num_steps, 1, self._resolution, self._resolution], dtype=self.output_type)
        for batch_elem_idx in range(batch_size):
            for step_idx in range(num_steps):
                output_imgs[batch_elem_idx, step_idx] = self.render_state_vec(state_vec_batch[batch_elem_idx, step_idx])
        return np.asarray(output_imgs, dtype=np.float32)

    def render(self, mode='human'):
        # render an image capturing the current state of the environment
        human_res = 640

        scale = 2 * (human_res // self._resolution) if mode == 'human' else 1
        render_res = int(self._resolution * scale)
        object_r = int(self._ball_r * scale)
        agent_sz = int(self._agent_size * scale)

        img = self.render_state(self._state, render_res, scale, object_r, agent_sz)

        if mode == 'human':
            img_human = cv2.resize(img, (human_res, human_res), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('Bouncing Balls', img_human[0])

        return img

    def normalize_vec(self, vec_data):
        # returns a normalized version of the data (only for vector-valued data)
        return (vec_data * 2 / self._resolution) - 1.0    # scale position data -1...1

    def unnormalize_vec(self, vec_data):
        # scales data back to original range (only for vector-valued data)
        return (vec_data + 1) * self._resolution / 2.0

    @property
    def output_size(self):
        if self._render_output:
            return [1, self._resolution, self._resolution]
        else:
            output_size = self._num_balls * 2
            output_size += 2 if not self._hide_agent else 0
            output_size += 2 if not self._hide_target else 0
            return [output_size]

    @property
    def output_type(self):
        return float

    @property
    def render_shape(self):
        return [1, self._resolution, self._resolution]

    @property
    def n_balls(self):
        return self._num_balls

    @property
    def n_goals(self):
        return 1        # multi-goal not implemented yet
