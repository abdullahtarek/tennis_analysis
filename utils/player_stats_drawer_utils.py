import numpy as np
import cv2

def draw_player_stats(output_video_frames,player_stats):

    for index, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width=350
        height=230

        start_x = frame.shape[1]-400
        start_y = frame.shape[0]-500
        end_x = start_x+width
        end_y = start_y+height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        text = "     Player 1     Player 2"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        
        text = "avg. S. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        text = "avg. P. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return output_video_frames
