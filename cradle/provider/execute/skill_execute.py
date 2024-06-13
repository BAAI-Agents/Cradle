from typing import Dict, Any
from copy import deepcopy

from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider

logger = Logger()
memory = LocalMemory()
video_record = VideoRecordProvider()

class SkillExecuteProvider(BaseProvider):
    def __init__(self, *args, gm: Any, **kwargs):

        super(SkillExecuteProvider, self).__init__()
        self.gm = gm

    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs) -> Dict[str, Any]:

        params = deepcopy(memory.current_info)

        skill_steps = params.get("skill_steps", [])

        start_frame_id = video_record.get_current_frame_id()
        exec_info = self.gm.execute_actions(skill_steps)
        screen_shot_path = self.gm.capture_screen()
        end_frame_id = video_record.get_current_frame_id()

        logger.write(f"Execute skill steps by frame id ({start_frame_id}, {end_frame_id}).")

        res_params = {
            "exec_info": exec_info,
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "screen_shot_path": screen_shot_path
        }
        memory.update_info_history(res_params)

        del params

        return res_params