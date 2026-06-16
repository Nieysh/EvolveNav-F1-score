# from .mp3d_agent import MP3DAgent
import random

from .mp3d_agent_once_forward_cot_navigation import MP3DAgent

class R2RAgent(MP3DAgent):
    name = "r2r"

    def get_prompt(self, task, *args, **kwargs):
        if task == 'navigation':
            return self.get_navigation_prompt(*args, **kwargs)
        elif task == 'summarization':
            return self.get_summarization_prompt(*args, **kwargs)
        elif task == 'embodied_qa':
            return self.get_embodied_qa_prompt(*args, **kwargs)
        # newly added
        elif task == 'spatial_relation':
            return self.get_spatial_relation_prompt(*args, **kwargs)
        elif task == 'navigation_cot':
            return self.get_navigation_cot_prompt(*args, **kwargs)
        elif task == 'navigation_cot_decision':
            return self.get_navigation_cot_decision_prompt(*args, **kwargs)
        elif task == "navigation_once_forward_cot_navigation":
            return self.get_navigation_once_forward_cot_navigation_prompt(*args, **kwargs)
        elif task == 'navigation_self_refine':
            return self.get_navigation_self_refine_prompt(*args, **kwargs)
        elif task == 'navigation_self_select':
            return self.get_navigation_self_select_prompt(*args, **kwargs)
        else:
            raise NotImplementedError

    def get_navigation_prompt(self, instruction, hist_num, cand_num, cls_token):
        # Task
        prompt = '### Instruction: Navigate following the instruction. {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) if i > 0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += 'Compare the History and Instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location.\n'
        prompt += '### Output: {}'.format(cls_token)

        return prompt

    def get_summarization_prompt(self, instruction, hist_num, cand_num):
        # Task
        prompt = f'### Instruction: Predict the fine-grained instruction based on your previous history and current location. Fine-grained instructions contain commands for each individual step. \n'
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        if cand_num != 0:
            prompt += 'Following is the Observation, which contains panoramic views at your current location.\n'
            obs_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
            prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += 'Please generate the step-by-step instruction.\n'
        prompt += '### Answer: '

        return prompt

    def get_embodied_qa_prompt(self, instruction, hist_num, cand_num):
        # Task
        prompt = f'### Instruction: answer the question. \n'
        # History
        if hist_num != 0:
            prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
            hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
            prompt += '### History: {}\n'.format(hist_text)
        # Observation
        if cand_num != 0:
            prompt += 'Following is the Observation, which contains panoramic views at your current location.\n'
            obs_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
            prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += '### Question: {}\n'.format(instruction)
        prompt += '### Answer: '

        return prompt

    # newly added
    def get_spatial_relation_prompt(self, cand_num, QA_landmark, query_cand_id):
        # if cand_num >= 2:
        #     # Task
        #     prompt = f'### Instruction: Please answer the question based on the observation.\n'
        #     # Observation
        #     prompt += 'Following is the Observation, which includes multiple views that can be observed from the present location.\n'
        #     obs_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
        #     prompt += '### Candidate: {}\n'.format(obs_text)
        #     # Question
        #     prompt += 'Following is the Question, which includes two landmarks that appear in the observation.\n'
        #     prompt += f'### Question: What is the spatial relationship of the {QA_landmark_pairs[1]} to {QA_landmark_pairs[0]}?\n'
        #     # Option
        #     prompt += 'Following is the Option, which includes 6 possible spatial relations between the two landmarks. First, check whether the two landmarks are at the similar level height (upon or under), and if yes, check the horizontal position relation between the two landmarks (front, rear, right, left).\n'
        #     prompt += '### Option: A. front, B. rear, C. right, D. left, E. upon, F. under\n'
        #     # Output Hint
        #     prompt += 'Please choose the correct option.\n'
        #     prompt += '### Answer: '
        # elif cand_num == 1:
        #     # Task
        #     prompt = f'### Instruction: Please answer the question based on the observation.\n'
        #     # Observation
        #     prompt += 'Following is the Observation, which includes a view that can be observed from the present location.\n'
        #     obs_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
        #     prompt += '### Candidate: {}\n'.format(obs_text)
        #     # Question
        #     prompt += 'Following is the Question, which includes a landmark that appear in the observation.\n'
        #     prompt += f'### Question: What is the spatial relationship of the {QA_landmark_pairs[0]} in the candidate?\n'
        #     # Option
        #     prompt += 'Following is the Option, which includes 6 possible spatial relations. First, check whether the landmark is at the similar level height as you(upon or under), and if yes, check the horizontal position relation between you and the landmark (front, rear, right, left).\n'
        #     prompt += '### Option: A. front, B. rear, C. right, D. left, E. upon, F. under\n'
        #     # Output Hint
        #     prompt += 'Please choose the correct option.\n'
        #     prompt += '### Answer: '
        # else:
        #     print("There is no cand that has landmark! Should handle this situation")
        #     raise NotImplementedError
        # Task
        prompt = f'### Instruction: Please answer the question based on the observation.\n'
        # Observation
        prompt += 'Following is the Observation, which includes multiple views that can be observed from the present location. Cand (0) is your current facing direction.\n'
        obs_text = ' '.join(['({}) <cand>'.format(i) for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Question
        prompt += 'Following is the Question, which includes landmarks that appear in a candidate.\n'
        prompt += f'### Question: What is the spatial relationship of the {QA_landmark} in Cand ({query_cand_id}) to your current facing direction?\n'
        # Option
        prompt += 'Following is the Option, which includes 6 possible spatial relations.\n' # First, check whether the landmarks are at the similar level height to you (upon or under), and if yes, check the horizontal position relation between you and the landmarks (front, rear, right, left).\n'
        prompt += '### Option: A. front, B. rear, C. right, D. left, E. upon, F. under\n'
        # Output Hint
        prompt += 'Please choose the correct option.\n'
        prompt += '### Answer: '

        return prompt

    def get_navigation_cot_prompt(self, instruction, hist_num, cand_num, cand_landmarks, nav_vpids, cand_masks):

        # Task
        prompt = '### Instruction: Navigate following the instruction. {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        if self.args.add_cand_landmark:
            nav_vpids = [nav_vpids[i] for i in range(len(nav_vpids)) if cand_masks[i]]
            obs_text = ' '.join(['(0) stop'] + [f"({i}) <cand> [{', '.join(cand_landmarks[nav_vpids[i]][:2])}]" for i in range(cand_num) if (i > 0)])
        else:
            obs_text = ' '.join(['({}) <cand>'.format(i) if i > 0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        # prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: According the Instruction, output your Long-term Goal, which is the final sub-instruction or the desitnation and the target object.\n- Short-term Goal: Compare the History and Instruction to infer your current progress, then output your Short-term Goal, which is an action chosen from your current candidates.\n- Spatial Relation: According to Candidate, infer the spatial relation of landmarks of each candidates to you. First, check whether the two landmarks are at the similar level height (upon or under), and if yes, check the horizontal position relation between the two landmarks (front, rear, right, left).\n- Common Sense: According to Long-term Goal, Short-term Goal, and Spatial Relation, infer which candidate fits the Short-term Goal and is likely to lead to the Long-term Goal.'
        # prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: According the Instruction, output your Long-term Goal, which is the final sub-instruction or the desitnation and the target object.\n- Short-term Goal: Compare the History and Instruction to infer your current progress, then output your Short-term Goal, which is an action chosen from your current candidates.\n- Reasoning: According to the Long-term Goal and the Short-term Goal, infer what observation may fit the Short-term Goal and is likely to lead to the Long-term Goal.'
        prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: Final sub-instruction or the desitnation and the target object.\n- Short-term Goal: The correct action chosen from your current candidates.\n- Reasoning: Infer what observation may fit the Short-term Goal and lead to the Long-term Goal.\n'
        prompt += '### Navigational Reasoning: \n'

        return prompt

    def get_navigation_cot_decision_prompt(self, instruction, hist_num, cand_num, cls_token, cot_input, cand_landmarks, nav_vpids, cand_masks):
        # Task
        prompt = '### Instruction: Navigate following the instruction. {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        if self.args.add_cand_landmark:
            nav_vpids = [nav_vpids[i] for i in range(len(nav_vpids)) if cand_masks[i]]
            obs_text = ' '.join(['(0) stop'] + [f"({i}) <cand> [{', '.join(cand_landmarks[nav_vpids[i]][:2])}]" for i in range(cand_num) if (i > 0)])
        else:
            obs_text = ' '.join(['({}) <cand>'.format(i) if i > 0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Navigation cot
        # prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: According the Instruction, output your Long-term Goal, which is the final sub-instruction or the desitnation and the target object.\n- Short-term Goal: Compare the History and Instruction to infer your current progress, then output your Short-term Goal, which is an action chosen from your current candidates.\n- Spatial Relation: According to Candidate, infer the spatial relation of landmarks of each candidates to you. First, check whether the two landmarks are at the similar level height (upon or under), and if yes, check the horizontal position relation between the two landmarks (front, rear, right, left).\n- Common Sense: According to Long-term Goal, Short-term Goal, and Spatial Relation, infer which candidate fits the Short-term Goal and is likely to lead to the Long-term Goal.'
        # prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: According the Instruction, output your Long-term Goal, which is the final sub-instruction or the desitnation and the target object.\n- Short-term Goal: Compare the History and Instruction to infer your current progress, then output your Short-term Goal, which is an action chosen from your current candidates.\n- Reasoning: According to the Long-term Goal and the Short-term Goal, infer what observation may fit the Short-term Goal and is likely to lead to the Long-term Goal.'
        prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: Final sub-instruction or the desitnation and the target object.\n- Short-term Goal: The correct action chosen from your current candidates.\n- Reasoning: Infer what observation may fit the Short-term Goal and lead to the Long-term Goal.\n'
        prompt += f'### Navigational Reasoning: {cot_input}'
        # Output Hint
        prompt += 'Make Action Decision according to Navigational Reasoning: Select the correct candidate to go to the target location.\n'
        prompt += '### Output: {}'.format(cls_token)

        return prompt

    def get_navigation_once_forward_cot_navigation_prompt(self, instruction, hist_num, cand_num, cand_landmarks, nav_vpids, cand_masks, cls_token, land_token=None, dir_token=None):
        # Task
        prompt = '### Instruction: Navigate following the instruction. {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation

        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        if self.args.add_cand_landmark:
            nav_vpids = [nav_vpids[i] for i in range(len(nav_vpids)) if cand_masks[i]]
            obs_text = ' '.join(['(0) stop'] + [f"({i}) <cand> [{', '.join(cand_landmarks[nav_vpids[i]][:2])}]" for i in range(cand_num) if (i > 0)])
            # print(f"obs_text: {obs_text}")
        else:
            obs_text = ' '.join(['({}) <cand>'.format(i) if i > 0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        if self.args.remove_qa_prompt_v2:
            prompt += 'Compare the History and Instruction to infer your current progress, and then select the correct direction from the candidates to go to the target location.\n'
        # prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: According the Instruction, output your Long-term Goal, which is the final sub-instruction or the desitnation and the target object.\n- Short-term Goal: Compare the History and Instruction to infer your current progress, then output your Short-term Goal, which is an action chosen from your current candidates.\n- Spatial Relation: According to Candidate, infer the spatial relation of landmarks of each candidates to you. First, check whether the two landmarks are at the similar level height (upon or under), and if yes, check the horizontal position relation between the two landmarks (front, rear, right, left).\n- Common Sense: According to Long-term Goal, Short-term Goal, and Spatial Relation, infer which candidate fits the Short-term Goal and is likely to lead to the Long-term Goal.'
        # prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: According the Instruction, output your Long-term Goal, which is the final sub-instruction or the desitnation and the target object.\n- Short-term Goal: Compare the History and Instruction to infer your current progress, then output your Short-term Goal, which is an action chosen from your current candidates.\n- Reasoning: According to the Long-term Goal and the Short-term Goal, infer what observation may fit the Short-term Goal and is likely to lead to the Long-term Goal.'
        if self.args.cot_summarization:
            if self.args.cot_v4:
                if self.args.mlm:
                    land_token_list = [land_token[0] for _ in range(self.args.land_token_region_length)]
                    land_token_region = "".join(land_token_list)
                    # prompt += 'Decide the action and generate the navigational reasoning with the format like: I should go to an observation with [table, chair, carpet, windows, vase] to the left of me.\n'
                    prompt += 'Generate the navigational reasoning and decide the action according to the instruction, history, and candidate.\n'
                    #prompt += '- Navigational Reasoning: I should ' + '{} '.format(dir_token[0])+ '{} '.format(dir_token[1])+ 'to an observation with ' +'{}, '.format(land_token[0]) + \
                    #          '{}, '.format(land_token[1]) + '{}, '.format(land_token[2]) + '{}, '.format(land_token[3]) \
                    #            +'{}'.format(land_token[4]) +  '.\n'

                    prompt += '- Navigational Reasoning: I should ' + '{}'.format(dir_token[0])+ '{} '.format(dir_token[1])+ 'to an observation with ' \
                                + land_token_region +  '.\n'
                    prompt += '- Action Decision: {}.'.format(cls_token)

                else:
                    # prompt += 'Decide the action and generate the navigational reasoning with the format like: I should go to an observation with [table, chair, carpet, windows, vase] to the left of me.\n'
                    # prompt += 'Decide the action and generate the navigational reasoning.\n'
                    # prompt += '- Action Decision: {}'.format(cls_token) + '.' + '\n'
                    # prompt += '- Navigational Reasoning: '
                    if self.args.remove_qa_prompt_v2:
                        prompt += 'Also generate the navigational reasoning of which observation (candidate) you should choose now by describing its landmarks and direction.\n'
                        prompt += '### Output: {}'.format(cls_token) + '.' + '\n'
                        prompt += '### Navigational Reasoning: '
                    else:
                        prompt += 'Decide the action and generate the navigational reasoning.\n'
                        prompt += '- Action Decision: {}'.format(cls_token) + '.' + '\n'
                        prompt += '- Navigational Reasoning: '
            elif self.args.cot_first_in_gt:
                prompt += 'Generate the navigational reasoning and decide the action.\n'
                prompt += '- Navigational Reasoning: '
            elif self.args.action_first_in_gt:
                prompt += 'Decide the action and generate the navigational reasoning.\n'
                prompt += '- Action Decision: '
            else:
                prompt += 'Generate the output which contains Navigational Reasoning and Action Decision: \n'
        else:
            prompt += 'Generate Navigational Reasoning of the following aspects:\n- Long-term Goal: Final sub-instruction or the desitnation and the target object.\n- Short-term Goal: The correct action chosen from your current candidates.\n- Reasoning: Infer what observation may fit the Short-term Goal and lead to the Long-term Goal.\n- Action Decision: The correct candidate.\n'

        if not self.args.cot_summarization:
            prompt += '### Navigational Reasoning: \n'

        return prompt

    def get_navigation_self_refine_prompt(self, instruction, hist_num, cand_num, cand_landmarks, nav_vpids, cand_masks, neg_cot):
        # Task
        prompt = '### Instruction: Navigate following the instruction. {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        if self.args.add_cand_landmark:
            nav_vpids = [nav_vpids[i] for i in range(len(nav_vpids)) if cand_masks[i]]
            obs_text = ' '.join(['(0) stop'] + [f"({i}) <cand> [{', '.join(cand_landmarks[nav_vpids[i]][:2])}]" for i in range(cand_num) if (i > 0)])
            # print(f"obs_text: {obs_text}")
        else:
            obs_text = ' '.join(['({}) <cand>'.format(i) if i > 0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += 'Identify the mistakes in the given output and generate the correct one.\n'
        prompt += f'Wrong:\n{neg_cot}\n'
        prompt += 'Correct:\n'

        return prompt

    def get_navigation_self_select_prompt(self, instruction, hist_num, cand_num, cand_landmarks, nav_vpids, cand_masks, cot_pair):
        # Task
        prompt = '### Instruction: Navigate following the instruction. {} \n'.format(instruction)
        # History
        prompt += 'Following is the History, which contains the visual information of your previous decisions.\n'
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        hist_text = ' '.join(['({}) <hist>'.format(i) for i in range(hist_num)])
        prompt += '### History: {}\n'.format(hist_text)
        # Observation
        prompt += 'Following is the Candidate, which contains several directions you can go to at the current position, candidate (0) is stop.\n'
        if self.args.add_cand_landmark:
            nav_vpids = [nav_vpids[i] for i in range(len(nav_vpids)) if cand_masks[i]]
            obs_text = ' '.join(['(0) stop'] + [f"({i}) <cand> [{', '.join(cand_landmarks[nav_vpids[i]][:2])}]" for i in range(cand_num) if (i > 0)])
            # print(f"obs_text: {obs_text}")
        else:
            obs_text = ' '.join(['({}) <cand>'.format(i) if i > 0 else '(0) stop' for i in range(cand_num)])
        prompt += '### Candidate: {}\n'.format(obs_text)
        # Output Hint
        prompt += f"Choose the correct one from the given two outputs.\n"
        prompt += f'Output1:\n{cot_pair[0]}\n'
        prompt += f'Output2:\n{cot_pair[1]}\n'
        prompt += 'Selection:\n'

        return prompt

class R2RAugAgent(R2RAgent):
    name = "r2r_aug"