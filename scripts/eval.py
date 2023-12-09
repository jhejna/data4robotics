import argparse
import time
import hydra
import numpy as np
import torch
import yaml
import os
import json
import cv2

import robots

import datetime


def convert_obs(obs, transform):
    img = torch.from_numpy(obs["agent_image"]).float().permute((2, 0, 1)) / 255
    img = transform(img)[None].cuda()

    state = np.concatenate((obs["state"]["ee_pos"], obs["state"]["ee_quat"], obs["state"]["gripper_pos"]))
    state = torch.from_numpy(state)[None].cuda()

    return dict(state=state, img=img)

def predict(agent, obs, mean, std):
    with torch.no_grad():
        ac = agent.get_actions(obs["img"], obs["state"])
    ac = ac[0,-1].cpu().numpy().astype(np.float32)
    ac = np.clip(ac * std + mean, -1, 1)    # denormalize the actions
    return ac


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,required=True)
    parser.add_argument("--normalization-path", type=str, default="scripts/blue_mug_dataset_statistics.json")
    parser.add_argument("--video-save-path", type=str, default=None)
    parser.add_argument("--config", default="/scr/jhejna/robot-lightning/configs/iliad_franka_orca.yaml")
    args = parser.parse_args()

    agent_path = os.path.expanduser(os.path.dirname(args.checkpoint))
    model_name = args.checkpoint.split('/')[-1]


    with open(os.path.join(agent_path, "agent_config.yaml"), "r") as f:
        config_yaml = f.read()
        agent_config = yaml.safe_load(config_yaml)
    with open(os.path.join(agent_path, "obs_config.yaml"), "r") as f:
        config_yaml = f.read()
        obs_config = yaml.safe_load(config_yaml)
    
    with open(args.normalization_path, 'r') as json_file:
        json_data = json.load(json_file)
        mean = np.array(json_data["mean"], dtype=np.float32)
        std = np.array(json_data["std"], dtype=np.float32)

    agent = hydra.utils.instantiate(agent_config)
    save_dict = torch.load(os.path.join(agent_path, model_name), map_location="cpu")
    agent.load_state_dict(save_dict['model'])
    agent = agent.cuda()
    agent.eval()

    transform = hydra.utils.instantiate(obs_config["transform"])

    with open(args.config, "r") as f:
        robot_config = yaml.load(f, Loader=yaml.Loader)
    env = robots.RobotEnv(**robot_config)

    while True:

        input("Press [Enter] to start.")

        obs = env.reset()
        time.sleep(3.0)
        done = False

        # do rollout
        images = [obs["agent_image"]]
        t = 0
        while t < 120 and not done:

            # get action
            action = predict(agent, convert_obs(obs, transform), mean, std)

            # perform environment step
            obs, _, done, _ = env.step(action)
            images.append(obs["agent_image"])
            t += 1

            if done:
                break

        # save video
        if args.video_save_path is not None:
            os.makedirs(args.video_save_path, exist_ok=True)
            # Create the directory for this trajectory
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            user_input = input("Was success or discard? [y/n/d]")
            if user_input == "d":
                continue
            elif user_input == "y":
                success = True
            else:
                success = False

            ckpt_tag = os.path.basename(os.path.normpath(args.checkpoint))
            checkpoint_eval_path = os.path.join(args.video_save_path, ckpt_tag)
            os.makedirs(checkpoint_eval_path, exist_ok=True)
            save_path = os.path.join(
                checkpoint_eval_path,
                f"{curr_time}_success-{success}.mp4",
            )
            video = np.stack(images)[:, -1, :, :, :]
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (256, 256))
            for img in video:
                writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            writer.release()

    

