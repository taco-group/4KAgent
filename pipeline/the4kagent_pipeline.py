import os
import gc
import cv2
import json
import shutil
import random
import logging
from time import localtime, strftime
from pathlib import Path
from typing import Union, Optional

import torch
from PIL import Image

from . import prompts
from executor import executor, Tool
from llm import GPT4, AzureGPT, MiniMaxLLM, DepictQA, PerceptionVLMAgent, LlamaVisionAgent

from utils.img_tree import ImgTree
from utils.logger import get_logger
from utils.misc import sorted_glob
from utils.custom_types import *
from utils.restore_profile import *
from utils.expert_IQA_eval import compute_iqa, compute_iqa_metric_score, compute_iqa_metric_score_batch
from utils.expert_face_score import compute_face_scores
from utils.color_fix import (
    adain_color_fix,
    wavelet_color_fix,
    cv2_to_pil,
    pil_to_cv2,
)
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from utils.scorer import calculate_cos_dist, calculate_niqe
from .profile_loader import load_profile_config


class The4KAgent:
    """
    Args:
        input_path (Path): Path to the input image.
        output_dir (Path): Path to the output directory, in which a directory will be created.
        llm_config_path (Path, optional): Path to the config file of LLM. Defaults to Path("config.yml").
        evaluate_degradation_by (str, optional): The method of degradation evaluation, "depictqa", "gpt4v" or "enhanced_gpt4v". Defaults to "depictqa".
        with_retrieval (bool, optional): Whether to schedule with retrieval. Defaults to True.
        schedule_experience_path (Path | None, optional): Path to the experience hub. Defaults to Path( "memory/schedule_experience.json").
        with_reflection (bool, optional): Whether to reflect on the results of tools. Defaults to True.
        reflect_by (str, optional): The method of reflection on results of tools, "depictqa" or "gpt4v". Defaults to "depictqa".
        with_rollback (bool, optional): Whether to roll back when failing in one subtask. Defaults to True.
        silent (bool, optional): Whether to suppress the console output. Defaults to False.
    """

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        llm_config_path: Path = Path("config.yml"),
        with_retrieval: bool = True,
        schedule_experience_path: Optional[Path] = Path("memory/schedule_experience.json"),
        with_reflection: bool = True,
        # with_rollback: bool = False,
        silent: bool = False,
        tool_run_gpu_id: Optional[int] = None,
        perception_agent_run_gpu_id: Optional[int] = None,
        profile_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        # paths
        self._prepare_dir(input_path, output_dir)
        # state
        self._init_state()
        # config
        self._config(
            with_retrieval,
            with_reflection,
            # with_rollback,
            tool_run_gpu_id,
            profile_name,
            llm_provider
        )
        # components
        self._create_components(llm_config_path, schedule_experience_path, silent)
        # constants
        self._set_constants()
        

    def _init_state(self) -> None:
        self.plan: list[Subtask] = []
        self.work_mem: dict = {
            "plan": {"initial": [], "adjusted": [
                # {
                #     "failed": [...] + [...],
                #     "new": [...] + [...]
                # }
            ]},
            "execution_path": {"subtasks": [], "tools": []},
            "n_invocations": 0,
            "tree": {
                "img_path": str(self.img_tree_dir / "0-img" / "input.png"),
                "best_descendant": None,
                "children": {
                    # `subtask1`: {
                    #     "best_tool": ...,
                    #     "tools": {
                    #         `tool1`: {
                    #             "degradation": ...,
                    #             "severity": ...,
                    #             "img_path": ...,
                    #             "best_descendant": ...,
                    #             "children": {...}
                    #         },
                    #         ...
                    #     }
                    # }
                },
            },
        }
        self.cur_node = self.work_mem["tree"]
        self.image_description = ""
        self.face_list = []
        

    def _config(
        self,
        with_retrieval: bool,
        with_reflection: bool,
        # with_rollback: bool,
        tool_run_gpu_id: Optional[int],
        profile_name: Optional[str] = None,
        llm_provider: Optional[str] = None,
    ) -> None:
        # extract profile
        self.profile_name = profile_name or "FastGen4K_P"
        self.profile = load_profile_config(self.profile_name)
        
        # evaluator
        self.evaluate_degradation_by = self.profile.get("PerceptionAgent", "llama_vision")
        self.reflect_by = self.profile.get("Reflection", "hpsv2+metric")
        assert self.evaluate_degradation_by in {"gpt4v", "depictqa", "vlmagent", "llama_vision"}
        assert self.reflect_by in {"hpsv2", "hpsv2+metric"}
        self.perception_agent_seed = self.profile.get("PerceptionAgent_Seed", 1994)
        
        # upscaling & 4k upscaling related
        self.upscale_4K = self.profile.get("Upscale4K", True)
        self.scale_factor = self.profile.get("ScaleFactor", None)
        self.require_sr_size = self.profile.get("require_sr_size", 300)
        
        # restoration options
        self.restore_option = self.profile.get("RestoreOption", None)
        self.restore_perference = self.profile.get("RestorePerference", None)
        
        # face restoration
        self.face_restoration = self.profile.get("FaceRestore", False)
        # old photo restoration
        self.old_photo_restoration = self.profile.get("OldPhotoRestoration", False)
        # other degradation
        self.brightening = self.profile.get("Brightening", False)
        
        # user define plan
        self.user_define = self.profile.get("User_Define", False)
        self.user_define_plan = self.profile.get("User_Define_Plan", None)
        
        # rollback
        # self.with_rollback = with_rollback
        self.with_rollback = self.profile.get("with_rollback", True)

        self.img_type = self.profile.get("ImgType", "General") or "General"
        self.with_retrieval = with_retrieval
        self.with_reflection = with_reflection
        self.tool_run_gpu_id = tool_run_gpu_id
        
        self.fast_4k = self.profile.get("Fast4K", False)
        self.fast4k_side_thres = self.profile.get("Fast4kSideThres", 1024)

        self.llm_provider = llm_provider or "gpt4"

        self.project_root = Path(__file__).resolve().parent.parent # 4kagent dir path
        
        
    def _create_components(
        self,
        llm_config_path: Path,
        schedule_experience_path: Optional[Path],
        silent: bool,
    ) -> None:
        # logging setup
        self.qa_logger = get_logger(
            logger_name="4KAgent QA",
            log_file=self.qa_path,
            console_log_level=logging.WARNING,
            file_format_str="%(message)s",
            silent=silent,
        )
        workflow_format_str = "%(asctime)s - %(levelname)s\n%(message)s\n"
        self.workflow_logger: logging.Logger = get_logger(
            logger_name="4KAgent Workflow",
            log_file=self.workflow_path,
            console_format_str=workflow_format_str,
            file_format_str=workflow_format_str,
            silent=silent,
        )

        # perception agent
        print(f"[Evaluation VLM] model: {self.evaluate_degradation_by}")
        if self.evaluate_degradation_by in {"vlmagent", "depictqa"}:
            self.perception_agent = PerceptionVLMAgent(
                seed=self.perception_agent_seed,
                logger=self.qa_logger,
                silent=silent,
                system_message=prompts.updated_perception_system_message,
            )
        elif self.evaluate_degradation_by == "llama_vision":
            self.perception_agent = LlamaVisionAgent(
                seed=self.perception_agent_seed,
                logger=self.qa_logger,
                silent=silent,
                system_message=prompts.updated_perception_system_message,
            )
        
        # language models
        if self.llm_provider == "minimax":
            self.gpt4 = MiniMaxLLM(
                config_path=llm_config_path,
                logger=self.qa_logger,
                silent=silent,
                system_message=prompts.system_message,
            )
        elif self.llm_provider == "azure":
            self.gpt4 = AzureGPT(
                config_path=llm_config_path,
                logger=self.qa_logger,
                silent=silent,
                system_message=prompts.system_message,
            )
        else:
            self.gpt4 = GPT4(
                config_path=llm_config_path,
                logger=self.qa_logger,
                silent=silent,
                system_message=prompts.system_message,
            )
        # self.gpt4 = AzureGPT(
        #     config_path=llm_config_path,
        #     logger=self.qa_logger,
        #     silent=silent,
        #     system_message=prompts.system_message,
        # )
        
        self.depictqa = None
        if self.evaluate_degradation_by == "depictqa" or self.reflect_by == "depictqa":
            self.depictqa = DepictQA(logger=self.qa_logger, silent=silent)

        # face restore
        self.face_helper = FaceRestoreHelper(
            upscale_factor=1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            model_rootpath=str(self.project_root / "executor/face_restoration/tools/GFPGAN/gfpgan/weights") # Checkpoints will be downloaded automatically
        )
        
        # experience
        if self.with_retrieval:
            assert schedule_experience_path is not None, "Experience should be provided."
            with open(schedule_experience_path, "r") as f:
                self.schedule_experience: str = json.load(f)["distilled"]

        # executor
        self.executor = executor
        
        #
        random.seed(0)
        
        
    def _set_constants(self) -> None:
        self.degra_subtask_dict: dict[Degradation, Subtask] = {
            "low resolution": "super-resolution",
            "low resolution_2x": "super-resolution_2x",
            "noise": "denoising",
            "motion blur": "motion deblurring",
            "defocus blur": "defocus deblurring",
            "haze": "dehazing",
            "rain": "deraining",
            "dark": "brightening",
            "jpeg compression artifact": "jpeg compression artifact removal",
            "low quality face": "face restoration",
            "old photo": "old_photo_restoration",
        }
        self.subtask_degra_dict: dict[Subtask, Degradation] = {
            v: k for k, v in self.degra_subtask_dict.items()
        }
        self.degradations = set(self.degra_subtask_dict.keys())
        self.subtasks = set(self.degra_subtask_dict.values())
        self.levels: list[Level] = ["very low", "low", "medium", "high", "very high"]
    

    def run(self, plan: Optional[list[Subtask]] = None, cache: Optional[Path] = None) -> None:
        self.workflow_logger.info(f"Running 4KAgent under the profile: {self.profile_name}")
        self.workflow_logger.info(f"Face Restoration: {self.face_restoration}")
        self.workflow_logger.info(f"Brightening: {self.brightening}")
        self.workflow_logger.info(f"Old Photo Restoration: {self.old_photo_restoration}")

        if self.old_photo_restoration:
            opr_dir, _, opr_toolbox = self._prepare_for_subtask("old_photo_restoration")
            opr_tool = opr_toolbox[0]
            opr_tool_dir = opr_dir / f"tool-{opr_tool.tool_name}"
            opr_output_dir = opr_tool_dir / "0-img"
            opr_output_dir.mkdir(parents=True, exist_ok=True)
            opr_tool(
                input_dir=Path(self.cur_node["img_path"]).parent,
                output_dir=opr_output_dir,
                silent=True,
                run_gpu_id=self.tool_run_gpu_id
            )
            self.cur_node["img_path"] = str(opr_output_dir / "output.png")
            self.executor._executed_subtask_cnt = 0

        if plan is not None:
            self.plan = plan.copy()
        elif self.user_define:
            self.plan = self.user_define_plan
            self.workflow_logger.info(f"Plan: {self.plan}")
        else:
            self.propose()
        
        while self.plan:
            success = self.execute_subtask(cache)
            if plan is None and self.with_rollback and not success:
                self.roll_back()
                self.reschedule()
        
        ### parallel computing
        ### TODO: we need to implement self.execute_subtask_distributed function
        # proposed_full_plan = self.plan.copy()
        # sr_subtasks = {"super-resolution", "super-resolution_2x"}
        # sr_index = next((i for i, subtask in enumerate(proposed_full_plan) if subtask in sr_subtasks), -1)
        # if sr_index == 0:
        #     while self.plan:
        #         success = self.execute_subtask(cache)
        #         if plan is None and self.with_rollback and not success:
        #             self.roll_back()
        #             self.reschedule()
        # elif sr_index > 0:
        #     for _ in range(sr_index):
        #         success = self.execute_subtask_distributed(cache)
        #         if not success and self.with_rollback:
        #             self.roll_back()
        #             self.reschedule()
        #     while self.plan:
        #         success = self.execute_subtask(cache)
        #         if self.plan is None and self.with_rollback and not success:
        #             self.roll_back()
        #             self.reschedule()
        # else:
        #     while self.plan:
        #         success = self.execute_subtask_distributed(cache)
        #         if self.plan is None and self.with_rollback and not success:
        #             self.roll_back()
        #             self.reschedule()
        
        self._record_res()
        

    def propose(self) -> None:
        """Sets the initial plan."""
        agenda = []

        def add_sr_tasks(target_factor: int):
            if target_factor == 2:
                agenda.append("super-resolution_2x")
            elif target_factor == 4:
                agenda.append("super-resolution")
            elif target_factor == 8:
                agenda.extend(["super-resolution_2x", "super-resolution"])
            elif target_factor == 16:
                agenda.extend(["super-resolution", "super-resolution"])

        if self.restore_option:
            explicit_tasks = self.restore_option.split('+')
            if 'super-resolution' in explicit_tasks:
                self.require_sr_size = 10 ** 6
            for task in explicit_tasks:
                assert task in self.degra_subtask_dict.values(), f"Invalid task: {task}"
                if task == "super-resolution":
                    img_shape = cv2.imread(self.cur_node["img_path"]).shape[:2]
                    target_factor = self.get_target_factor(img_shape)
                    self.workflow_logger.info(f"Target factor for super-resolution: {target_factor}")
                    if target_factor > 1:
                        add_sr_tasks(target_factor)
                else:
                    agenda.append(task)

            if "hpsv2" in self.reflect_by:
                self.image_description = self.get_image_description()

        else:
            if self.scale_factor is not None:
                self.require_sr_size = 10 ** 6

            if (self.evaluate_degradation_by not in ["vlmagent", "llama_vision"] 
                and "hpsv2" in self.reflect_by):
                self.image_description = self.get_image_description()

            evaluation = self.evaluate_degradation()
            agenda = self.extract_agenda(evaluation)

        if self.face_restoration or 'face restoration' in agenda:
            self.face_list = self.extract_face(self.cur_node["img_path"], self.faces_dir)
            self.face_helper.clean_all()
            agenda = [task for task in agenda if task != 'face restoration']

        if not self.brightening and 'brightening' in agenda:
            agenda.remove('brightening')

        plan = self.schedule(agenda)
        self.work_mem["plan"]["initial"] = plan.copy()
        self._dump_summary()
        self.workflow_logger.info(f"Plan: {plan}")
        self.plan = plan

        del self.perception_agent
        gc.collect()
        torch.cuda.empty_cache()


    def extract_face(self, input_path: Union[Path, str], res_path) -> None:
        in_path = str(input_path.resolve()) if isinstance(input_path, Path) else input_path
        self.face_helper.read_image(in_path)
        self.face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        face_paths = []
        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            save_folder = res_path / f'face_{idx:03d}'
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = save_folder / 'face.png'
            cv2.imwrite(str(save_path), cropped_face)
            face_paths.append(str(save_path))

        return face_paths
    

    def get_target_factor(self, img_shape) -> int:
        """
        Determine the desired upscaling factor based on the image size, the upscale_4K flag,
        the required SR size, and an external scale_factor parameter.

        When upscale_4K is False:
        - Follow the legacy logic: if the image's maximum side is less than require_sr_size,
            then use the external scale_factor (or default to 4 if scale_factor is None).
        - If the image is equal to or larger than require_sr_size, then no upscaling is needed (return 1).

        When upscale_4K is True:
        - Only upscale images with a maximum side less than 4096.
        - Compute the raw factor as 4096 / max_side.
        - If the computed factor is less than 4, force a minimum upscale of 2×.
        - Otherwise, discretize the raw value into one of the allowed factors: 4, 8, or 16.
        """
        max_side = max(img_shape)

        def compute_factor_to_4k(side_len: int) -> int: 
            # Compute the smallest factor in {2,4,8,16} to reach ≥ 4096
            for factor in [2, 4, 8, 16]:
                if side_len * factor >= 4000:
                    return factor
            return 16  # fallback (in case of very small image)
        
        if not self.upscale_4K:
            # Legacy behavior for non-4K upscaling:
            if max_side < self.require_sr_size:
                # If the image is smaller than the required SR size,
                # use the external scale_factor if provided, else default to 4× upscale.
                if self.scale_factor is None:
                    return 4
                if self.scale_factor in (2, 4, 8, 16):
                    return self.scale_factor
                else:
                    raise ValueError("scale_factor must be one of 2, 4, 8, or 16")
            else:
                return 1
        else:
            if max_side >= 4000:
                return 1
            return compute_factor_to_4k(max_side)


    def extract_agenda(self, evaluation: Union[list[tuple[Degradation, Level]], list[Degradation]]) -> list[Subtask]:
        agenda = []
        img_shape = cv2.imread(self.cur_node["img_path"]).shape[:2]

        # Determine the target upscaling factor (allowed values: 1, 2, 4, 8, 16; where 1 means no SR task).
        target_factor = self.get_target_factor(img_shape)

        def _append_sr_tasks(factor: int) -> list[str]:
            """Helper to append appropriate super-resolution tasks based on factor."""
            factor_map = {
                2: ["super-resolution_2x"],
                4: ["super-resolution"],
                8: ["super-resolution_2x", "super-resolution"],
                16: ["super-resolution", "super-resolution"],
                # directly 16x experiments
            }
            return factor_map.get(factor, [])

        if self.evaluate_degradation_by in {"vlmagent", "llama_vision"}:
            agenda = [self.degra_subtask_dict[deg] for deg in evaluation]
            if target_factor > 1:
                agenda.extend(_append_sr_tasks(target_factor))
            random.shuffle(agenda)
        else:
            for degradation, severity in evaluation:
                if self.levels.index(severity) >= 2:  # "medium" or higher
                    agenda.append(self.degra_subtask_dict[degradation])
            if target_factor > 1:
                agenda.extend(_append_sr_tasks(target_factor))
            random.shuffle(agenda)
        
        return agenda


    def evaluate_degradation(self) -> list[tuple[Degradation, Level]]:
        """Perception stage: Evaluate the degradation of the input image.
        (motion blur, defocus blur, rain, haze, dark, noise, jpeg compression artifact).
        """
        eval_map = {
            "gpt4v": self.evaluate_degradation_by_gpt4v,
            "vlmagent": self.evaluate_degradation_by_vlmagent,
            "llama_vision": self.evaluate_degradation_by_llama_vision,
        }

        evaluator = eval_map.get(self.evaluate_degradation_by)
        if evaluator:
            evaluation = evaluator()
        else:
            evaluation = eval(self.depictqa(Path(self.cur_node["img_path"]), task="eval_degradation"))

        self.workflow_logger.info(f"Evaluation: {evaluation}")
        return evaluation


    def evaluate_degradation_by_gpt4v(self) -> list[tuple[Degradation, Level]]:
        def check_evaluation(evaluation: object):
            assert isinstance(evaluation, list), "Evaluation should be a list."
            rsp_degradations = set()
            for ele in evaluation:
                assert isinstance(
                    ele, dict
                ), "Each element in evaluation should be a dict."
                assert set(ele.keys()) == {
                    "degradation",
                    "thought",
                    "severity",
                }, f"Invalid keys: {ele.keys()}."
                degradation = ele["degradation"]
                rsp_degradations.add(degradation)
                severity = ele["severity"]
                assert severity in self.levels, f"Invalid severity: {severity}."
            assert rsp_degradations == self.degradations - {
                "low resolution"
            }, f"Invalid degradation: {rsp_degradations}."

        evaluation = eval(
            self.gpt4(
                prompt=prompts.gpt_evaluate_degradation_prompt,
                img_path=Path(self.cur_node["img_path"]),
                format_check=check_evaluation,
            )
        )
        evaluation = [(ele["degradation"], ele["severity"]) for ele in evaluation]
        return evaluation


    def get_image_description(self) -> str:
        iqa_scores_results, img_height, img_width = compute_iqa(self.cur_node["img_path"])
        self.workflow_logger.info(f"IQA scores: {iqa_scores_results}")
        
        if self.evaluate_degradation_by == "vlmagent":
            prompt = prompts.updated_perception_system_prompt.format(iqa_result=iqa_scores_results)
        else:
            prompt = prompts.llama_vision_agent_perception_no_brighten_system_message.format(iqa_result=iqa_scores_results)
        perception_inputs = self.perception_agent.prepare_inputs(image_path=[self.cur_node["img_path"]], 
                                                        prompts=[prompt])
        perception_output = self.perception_agent.perception(inputs=perception_inputs, max_new_tokens=1600)
        output = perception_output['image_description']
        self.workflow_logger.info(f"Image description: {perception_output['image_description']}")
        
        # Release memory
        del perception_inputs
        del perception_output
        gc.collect()
        torch.cuda.empty_cache()
        
        return output


    def evaluate_degradation_by_vlmagent(self) -> list[Degradation]:
        iqa_scores_results, img_height, img_width = compute_iqa(self.cur_node["img_path"])
        self.workflow_logger.info(f"IQA scores: {iqa_scores_results}")

        perception_inputs = self.perception_agent.prepare_inputs(
            image_path=[self.cur_node["img_path"]],
            prompts=[prompts.updated_perception_system_prompt.format(iqa_result=iqa_scores_results)]
        )
        perception_output = self.perception_agent.perception(inputs=perception_inputs, max_new_tokens=1600)

        self.image_description = perception_output.get('image_description', '')
        self.workflow_logger.info(f"Image description: {self.image_description}")

        degradations = perception_output.get('degradations', [])

        # Release memory
        torch.cuda.empty_cache()

        return degradations


    def evaluate_degradation_by_llama_vision(self) -> list[Degradation]:
        iqa_scores_results, img_height, img_width = compute_iqa(self.cur_node["img_path"])
        self.workflow_logger.info(f"IQA scores: {iqa_scores_results}")

        perception_prompt = prompts.llama_vision_agent_perception_no_brighten_system_message.format(
            iqa_result=iqa_scores_results
        )
        perception_inputs = self.perception_agent.prepare_inputs(
            image_path=[self.cur_node["img_path"]],
            prompts=[perception_prompt]
        )
        perception_output = self.perception_agent.perception(inputs=perception_inputs, max_new_tokens=1600)

        self.image_description = perception_output.get('image_description', '')
        self.workflow_logger.info(f"Image description: {self.image_description}")

        degradations = perception_output.get('degradations', [])

        # Release memory
        del perception_inputs
        del perception_output
        gc.collect()
        torch.cuda.empty_cache()

        return degradations


    def schedule(self, agenda: list[Subtask], ps: str = "") -> list[Subtask]:
        if len(agenda) <= 1:
            return agenda

        degradations = [self.subtask_degra_dict[subtask] for subtask in agenda]

        if degradations.count("low resolution") > 2:
            degradations = [d for d in degradations if d != "low resolution"]
            degradations.append("low resolution")

        if self.evaluate_degradation_by in {"vlmagent", "llama_vision"}:
            plan = self.schedule_updated_w_retrieval(degradations, agenda, self.image_description)
        else:
            if self.with_retrieval:
                plan = self.schedule_w_retrieval(degradations, agenda, ps)
            else:
                plan = self.schedule_wo_retrieval(degradations, agenda, ps)
                
        return plan
    

    def schedule_w_retrieval(
        self, degradations: list[Degradation], agenda: list[Subtask], ps: str
    ) -> list[Subtask]:
        def check_order(schedule: object):
            assert isinstance(schedule, dict), "Schedule should be a dict."
            assert set(schedule.keys()) == {"thought", "order"}, \
                f"Invalid keys: {schedule.keys()}."
            order = schedule["order"]
            assert set(order) == set(agenda), \
                f"{order} is not a permutation of {agenda}."

        formated_prompt = prompts.schedule_w_retrieval_prompt.format(
                degradations=degradations, agenda=agenda, 
                experience=self.schedule_experience
            ) + ps
        self.workflow_logger.info(f"prompt: {formated_prompt}")
        schedule = self.gpt4(
            prompt=prompts.schedule_w_retrieval_prompt.format(
                degradations=degradations, agenda=agenda, 
                experience=self.schedule_experience
            ) + ps,
            format_check=check_order,
        )
        self.workflow_logger.info(f"Response: {schedule}")
        schedule = eval(schedule)
        self.workflow_logger.info(f"Insights: {schedule['thought']}")
        return schedule["order"]
    

    def reason_to_schedule(
        self, degradations: list[Degradation], agenda: list[Subtask]
    ) -> str:
        insights = self.gpt4(
            prompt=prompts.reason_to_schedule_prompt.format(
                degradations=degradations, agenda=agenda
            ),
        )
        self.workflow_logger.info(f"Insights: {insights}")
        return insights
    
    
    def schedule_wo_retrieval(
        self, degradations: list[Degradation], agenda: list[Subtask], ps: str
    ) -> list[Subtask]:
        insights: str = self.reason_to_schedule(degradations, agenda)

        def check_order(order: object):
            assert isinstance(order, list), "Order should be a list."
            assert set(order) == set(agenda), f"{order} is not a permutation of {agenda}."

        order = self.gpt4(
            prompt=prompts.schedule_wo_retrieval_prompt.format(
                degradations=degradations, agenda=agenda, insights=insights
            ) + ps,
            format_check=check_order,
        )
        return eval(order)
    

    def schedule_updated_w_retrieval(
        self, degradations: list[Degradation], agenda: list[Subtask], image_description: str
    ) -> list[Subtask]:
        def check_order(schedule: object):
            assert isinstance(schedule, dict), "Schedule should be a dict."
            # assert set(schedule.keys()) == {"thought", "order"}, \
            #     f"Invalid keys: {schedule.keys()}."
            order = schedule["order"]
            assert set(order) == set(agenda), \
                f"{order} is not a permutation of {agenda}."

        formated_prompt = prompts.schedule_updated_w_retrieval_prompt.format(
                image_description=image_description,
                degradations=degradations, agenda=agenda, 
                experience=self.schedule_experience
            )
        self.workflow_logger.info(f"prompt: {formated_prompt}")
        if self.evaluate_degradation_by == "llama_vision":
            schedule_inputs = self.perception_agent.prepare_inputs(image_path=[self.cur_node["img_path"]], 
                                                        prompts=[prompts.llama_vision_agent_plan_system_message.format(
                                                        image_description=self.image_description,
                                                        degradations=degradations,
                                                        tasks=agenda,
                                                        experience=self.schedule_experience
                                                        )])
            schedule = self.perception_agent.plan(inputs=schedule_inputs, agenda=agenda, max_new_tokens=1600)
            self.workflow_logger.info(f"Response: {schedule}")
            schedule["order"] = schedule["plan"]
            # Release memory
            del schedule_inputs
            gc.collect()
            torch.cuda.empty_cache()
        else:
            schedule = self.gpt4(
                prompt=prompts.schedule_updated_w_retrieval_prompt.format(
                    image_description=image_description,
                    degradations=degradations, agenda=agenda, 
                    experience=self.schedule_experience
                ),
                format_check=check_order,
            )
            self.workflow_logger.info(f"Response: {schedule}")
            schedule = eval(schedule)
            self.workflow_logger.info(f"Insights: {schedule['thought']}")
        return schedule["order"]


    def tool_selection(self, subtask: Subtask, toolbox: list[Tool], max_side: int) -> list[Tool]:
        """Select appropriate tools from toolbox based on profile, subtask, and resolution."""

        def get_baseline_tools(subtask: str) -> list[str]:
            baseline_mapping = {
                "brightening": General_Baseline_Brightening_ToolName,
                "defocus deblurring": General_Baseline_Defocus_Deblurring_ToolName,
                "dehazing": General_Baseline_Dehazing_ToolName,
                "denoising": General_Baseline_Denoising_ToolName,
                "deraining": General_Baseline_Deraining_ToolName,
                "jpeg compression artifact removal": General_Baseline_Jpeg_Compression_Artifact_Removal_ToolName,
                "motion deblurring": General_Baseline_Motion_Deblurring_ToolName,
                "super-resolution": General_Baseline_SuperResolution_ToolName,
            }
            if subtask not in baseline_mapping:
                raise ValueError(f"Invalid subtask: {subtask}")
            return baseline_mapping[subtask]

        def get_profiled_sr_tools(subtask: str) -> list[str]:
            if subtask == "super-resolution":
                if self.restore_perference == "Fidelity":
                    return General_SuperResolution_Fidelity_ToolName
                else:
                    return General_SuperResolution_Perception_ToolName
            elif subtask == "super-resolution_2x":
                if self.restore_perference == "Fidelity":
                    return General_SuperResolution_2x_Fidelity_ToolName
                else:
                    return General_SuperResolution_2x_Perception_ToolName
            return []

        if "Baseline" in self.profile_name:
            tool_names = get_baseline_tools(subtask)
            return [tool for tool in toolbox if tool.tool_name in tool_names]

        # Fidelity/Perception preference for SR tasks
        if subtask in ["super-resolution", "super-resolution_2x"] and self.restore_perference in ["Fidelity", "Perception"]:
            tool_names = get_profiled_sr_tools(subtask)
            updated_toolbox = [tool for tool in toolbox if tool.tool_name in tool_names]
        else:
            updated_toolbox = toolbox

        # Fast mode filtering
        if "Fast" in self.profile_name and "super-resolution" in subtask and max_side >= self.fast4k_side_thres:
            self.workflow_logger.info("Fast mode: only keep fast tools (currently only for super-resolution)")
            updated_toolbox = [tool for tool in updated_toolbox if tool.tool_name != 'diffbir']

        return updated_toolbox
                

    def execute_subtask(self, cache: Optional[Path]) -> bool:
        """Invokes tools to try to execute the top subtask in `self.plan` on `self.cur_node["img_path"]`, the directory of which is "0-img". Returns success or not. Updates `self.plan` and `self.cur_node`. Generates a directory parallel to "0-img", containing multiple directories, each of which contains outputs of a tool.\n
        Before:
        ```
        .
        ├── 0-img
        │   └── {input_path}
        └── ...
        ```
        After:
        ```
        .
        ├── 0-img
        │   └── {input_path}
        ├── {subtask_dir}
        |   ├── {tool_dir} 1
        |   │   └── 0-img
        |   │       └── output.png
        |   ├── ...
        |   └── {tool_dir} n
        |       └── 0-img
        |           └── output.png
        └── ...
        ```
        """

        subtask = self.plan.pop(0)
        subtask_dir, degradation, toolbox = self._prepare_for_subtask(subtask)
        res_degra_level_dict: dict[str, list[Path]] = {}
        success = True

        # obtain image size & toolbox
        cur_img = cv2.imread(self.cur_node["img_path"])
        img_shape = cur_img.shape[:2]
        max_side = max(img_shape)
        updated_toolbox = self.tool_selection(subtask, toolbox, max_side)

        for tool in updated_toolbox:
            self.work_mem["n_invocations"] += 1
            # prepare directory
            tool_dir = subtask_dir / f"tool-{tool.tool_name}"
            output_dir = tool_dir / "0-img"
            output_dir.mkdir(parents=True)

            # invoke tool
            if cache is None:
                tool(
                    input_dir=Path(self.cur_node["img_path"]).parent,
                    output_dir=output_dir,
                    silent=True,
                    run_gpu_id=self.tool_run_gpu_id
                )
            else:
                dst_path = output_dir / "output.png"
                rel_path = dst_path.relative_to(self.img_tree_dir)
                src_path = cache / rel_path
                dst_path.symlink_to(src_path)
            output_path = sorted_glob(output_dir)[0]

            if self.with_reflection:
                self._record_tool_res(output_path)
                res_degra_level_dict.setdefault(self.reflect_by, []).append(output_path)
            else:
                best_tool_name = tool.tool_name
                # best_img_path = output_path
                res_degra_level = "none"
                self._record_tool_res(output_path)
                break

        else:
            best_img_path, best_img_score = self.evaluate_tool_result_onetime(res_degra_level_dict[self.reflect_by])
                
            best_tool_name = self._get_name_stem(best_img_path.parents[1].name)
            self.workflow_logger.info(f"Best tool: {best_tool_name}")
                
            res_degra_level = self.reflect_by
            rollback_score = 0.12 if self.reflect_by == "hpsv2" else 0.5
                
            if best_img_score < rollback_score:
                success = False
            elif len(self.face_list) > 0 and subtask == 'super-resolution':
                self.face_restore(best_img_path)
        
        # Global img color alignment on large size image
        output_img = cv2.imread(str(best_img_path))
        if "super-resolution" in subtask:
            output_max_side = max(output_img.shape[:2])
            if output_max_side >= 1024:
                self.workflow_logger.info("Applying global color alignment...")
                color_fix_pil = adain_color_fix(cv2_to_pil(output_img), cv2_to_pil(cur_img))
                color_fix_img = pil_to_cv2(color_fix_pil)
                cv2.imwrite(str(best_img_path), color_fix_img)

        self.cur_node["children"][subtask]["best_tool"] = best_tool_name
        self.cur_node = self.cur_node["children"][subtask]["tools"][best_tool_name]
        if self.with_rollback and not success:
            self.cur_node["best_descendant"] = str(best_img_path)
            done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
            self.work_mem["plan"]["adjusted"].append({
                "failed": f"{done_subtasks} + {self.plan}", "new": None
            })

        self._dump_summary()
        self._render_img_tree()
        self.workflow_logger.info(
            f"{subtask.capitalize()} result: "
            f"{self._img_nickname(self.cur_node['img_path'])} "
            f"with quality score {best_img_score}.")
            
        return success
    
    
    def evaluate_tool_result_onetime(self, candidates: list[Path]) -> tuple[Path, float]:
        if not candidates:
            raise ValueError("`candidates` is empty.")

        task_folder = candidates[0].parents[2] if len(candidates[0].parents) > 2 else candidates[0].parent
        candidates_tmp_dir = os.path.join(str(task_folder), "tmp")
        os.makedirs(candidates_tmp_dir, exist_ok=True)

        candidate_paths = []
        for i, cand in enumerate(candidates):
            tool_name = self._get_name_stem(cand.parents[1].name) if len(cand.parents) > 1 else f"tool_{i:02d}"
            new_filename = f"image_{tool_name}.png"
            new_path = os.path.join(candidates_tmp_dir, new_filename)
            shutil.copy(str(cand), new_path)
            candidate_paths.append(str(cand))

        if self.reflect_by == "hpsv2+metric":
            metric_scores = [compute_iqa_metric_score(p) for p in candidate_paths]
            # metric_scores = compute_iqa_metric_score_batch(candidate_paths)

        import hpsv2
        hps_scores = hpsv2.score(candidate_paths, self.image_description, hps_version="v2.1")
        hps_scores = [float(s) for s in hps_scores]

        if self.reflect_by == "hpsv2+metric":
            result = [h + m for h, m in zip(hps_scores, metric_scores)]
            out_file = os.path.join(candidates_tmp_dir, "result_scores_with_metrics.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                for cand, h, m, o in zip(candidates, hps_scores, metric_scores, result):
                    name = self._get_name_stem(cand.parents[1].name) if len(cand.parents) > 1 else "unknown"
                    f.write(f"image_{name}, HPSv2: {h:.6f}, Metric: {m:.6f}, Overall: {o:.6f}\n")
        else:
            result = hps_scores
            out_file = os.path.join(candidates_tmp_dir, "result_scores.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                for cand, s in zip(candidates, result):
                    name = self._get_name_stem(cand.parents[1].name) if len(cand.parents) > 1 else "unknown"
                    f.write(f"image_{name}, {s:.6f}\n")

        best_idx, best_score = max(enumerate(result), key=lambda x: x[1])
        best_image = candidates[best_idx]
        # Release memory
        del hpsv2
        del result
        gc.collect()
        torch.cuda.empty_cache()

        return best_image, float(best_score)
    

    def evaluate_tool_result_by_gpt4v(
        self, img_path: Path, degradation: Degradation
    ) -> Level:
        def check_tool_res_evaluation(evaluation: object):
            assert isinstance(evaluation, dict), "Evaluation should be a dict."
            assert set(evaluation.keys()) == {
                "thought",
                "severity",
            }, f"Invalid keys: {evaluation.keys()}."
            severity = evaluation["severity"]
            assert severity in self.levels, f"Invalid severity: {severity}."

        degra_level = eval(
            self.gpt4(
                prompt=prompts.gpt_evaluate_tool_result_prompt.format(
                    degradation=degradation
                ),
                img_path=img_path,
                format_check=check_tool_res_evaluation,
            )
        )["severity"]
        return degra_level
    

    def search_best_by_comp(self, candidates: list[Path]) -> Path:
        """Compares multiple images to decide the best one."""

        best_img = candidates[0]
        for i in range(1, len(candidates)):
            cur_img = candidates[i]
            self.workflow_logger.info(
                f"Comparing {self._img_nickname(best_img)} and {self._img_nickname(cur_img)}..."
            )

            choice = self.compare_quality(best_img, cur_img)

            if choice == "latter":
                best_img = cur_img
                self.workflow_logger.info(
                    f"{self._img_nickname(best_img)} is better."
                )
            elif choice == "former":
                self.workflow_logger.info(
                    f"{self._img_nickname(best_img)} is better."
                )
            else:  # neither; keep the former
                self.workflow_logger.info(
                    f"Hard to decide. Keeping {self._img_nickname(best_img)}."
                )
        self.workflow_logger.info(
            f"{self._img_nickname(best_img)} is selected as the best."
        )
        return best_img

    def compare_quality(self, img1: Path, img2: Path) -> str:
        if self.reflect_by == "gpt4v":
            choice = self.compare_quality_by_gpt4v(img1, img2)
        else:
            choice = self.depictqa(img_path=[img1, img2], task="comp_quality")
        return choice

    def compare_quality_by_gpt4v(self, img1: Path, img2: Path) -> str:
        def check_comparison(comparison: object):
            assert isinstance(comparison, dict), "Comparison should be a dict."
            assert set(comparison.keys()) == {
                "thought",
                "choice",
            }, f"Invalid keys: {comparison.keys()}."
            assert comparison["choice"] in {
                "former",
                "latter",
                "neither",
            }, f"Invalid choice: {comparison['choice']}."

        comparison: dict = eval(
            self.gpt4(
                prompt=prompts.gpt_compare_prompt,
                img_path=[img1, img2],
                format_check=check_comparison,
            )
        )
        return comparison["choice"]


    def face_restore(self, img_path: Path) -> None:
        # There is a tool list in face restoration toolbox, run all tools in the toolbox and compare the results
        # The best result is selected as the final result
        sub_task = 'face restoration'
        subtask_dir, toolbox = self._prepare_for_subtask_face_restore(sub_task)
        # Copy the input image to the subtask directory
        ori_image_path = subtask_dir / "ori"
        ori_image_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(img_path), ori_image_path / "input.png")
        # Extract face from the input image
        step_face_path = subtask_dir / "0-img"
        step_face_path.mkdir(parents=True, exist_ok=True)
        face_img_paths = self.extract_face(img_path, step_face_path)
        if len(face_img_paths) != len(self.face_list):
            return
        # Evaluate the face image
        evaluate_results = []
        
        for i in range(len(face_img_paths)):
            face_evaluations = []
            face_evaluation = self.evaluate_tool_result_face(face_img_paths[i], self.face_list[i])
            self.workflow_logger.info(
                f"Face: {i:03d} "
                f"{sub_task.capitalize()} result: "
                f"Origin Face "
                f"Score: {face_evaluation}.")
            face_evaluations.append({'path': face_img_paths[i], 'evaluation': face_evaluation})
            for tool in toolbox:
                # prepare directory
                tool_dir = subtask_dir / f"tool-face-{i}-{tool.tool_name}"
                output_dir = tool_dir / "0-img"
                output_dir.mkdir(parents=True)
                # invoke tool
                tool(input_dir=Path(face_img_paths[i]).parents[0],output_dir=output_dir,silent=True,)
                
                # Evaluate the result (weighted score, should be a dict with {'face_00': score, 'face_01': score, ...})
                out_face_path = str(output_dir / 'output.png')
                face_evaluation = self.evaluate_tool_result_face(out_face_path, self.face_list[i])
                self.workflow_logger.info(
                    f"Face: {i:03d} "
                    f"{sub_task.capitalize()} result: "
                    f"{tool.tool_name} "
                    f"Score: {face_evaluation}.")

                face_evaluations.append({'path': out_face_path, 'evaluation': face_evaluation})
            best_face_path = max(face_evaluations, key=lambda x: x['evaluation'])['path']
            evaluate_results.append(face_evaluations)
            self.face_helper.add_restored_face(cv2.imread(str(best_face_path)))
        
        # Merge the face images back to the original image
        self.face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=None)
        cv2.imwrite(str(img_path), restored_img)
        self.face_helper.clean_all()

        result_file = os.path.join(subtask_dir, "result_scores_faces.txt")
        with open(result_file, "w") as f:
            for evaluate_result in evaluate_results:
                for face_evaluation in evaluate_result:
                    short_path = face_evaluation['path'].split('/')[-3]
                    f.write(f"{short_path}, {face_evaluation['evaluation']}\n")
                f.write('\n')
        f.close()
        

    def evaluate_tool_result_face(self, restored_face: Path, ori_face: str) -> float:
        identity_score = calculate_cos_dist(str(restored_face), ori_face)
        nr_metric_score = compute_iqa_metric_score(str(restored_face))
        face_iqa_score = compute_face_scores(str(restored_face))
        Combined_score = face_iqa_score + nr_metric_score - ( identity_score / 1000 )
        return Combined_score


    def roll_back(self) -> None:
        # backtrack
        self._backtrack()
        step = 1
        while self._fully_expanded():
            self.workflow_logger.info(
                f"All execution paths from {self._img_nickname(self.cur_node['img_path'])} "
                f"lead to severe degradation.")
            self._set_best_desc()
            if self.cur_node != self.work_mem["tree"]:
                step += 1
                self._backtrack()
            else:
                break
        self.workflow_logger.info(
            f"Roll back for {step} step(s) "
            f"to {self._img_nickname(self.cur_node['img_path'])} "
            f"with agenda {self.plan}."
        )

        # compromise
        if self._fully_expanded():  # back to root
            self._to_best_desc(Path(self.cur_node["best_descendant"]))
            self.workflow_logger.info(
                "All execution paths from the input lead to severe degradation.\n"
                f"Compromise: jump to {self._img_nickname(self.cur_node['img_path'])} "
                f"with agenda {self.plan}."
            )
            assert not self._fully_expanded() or not self.plan, \
                "Invalid compromise: cannot go on or terminate."
        
        # check
        done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
        done_subtasks, plan = set(done_subtasks), set(self.plan)
        assert done_subtasks & plan == set(), \
            f"Invalid plan: {done_subtasks} & {plan} != ∅."
        assert done_subtasks | plan == set(self.work_mem["plan"]["initial"]), (
            f"Invalid plan: {done_subtasks} | {plan} != "
            f"{self.work_mem['plan']['initial']}.")
        

    def _fully_expanded(self) -> bool:
        return len(self.plan) == len(self.cur_node["children"])
    

    def _set_best_desc(self) -> None:
        candidates = [
            Path(subtask_res["tools"][subtask_res["best_tool"]]["best_descendant"])
            for subtask_res in self.cur_node["children"].values()
        ]
        self.workflow_logger.info("Searching for the best descendant...")
        if "hpsv2" in self.reflect_by:
            best_img_path, best_img_hpsv2_score = self.evaluate_tool_result_onetime(candidates)
            best_img_path = Path(best_img_path)
        else:
            best_img_path = self.search_best_by_comp(candidates)
        self.cur_node["best_descendant"] = str(best_img_path)


    def _to_best_desc(self, best_desc_path: Path):
        self.cur_node = self._img_path_to_node(best_desc_path)
        done_subtasks, _ = self._get_execution_path(best_desc_path)
        self.plan = list(set(self.plan) - set(done_subtasks))
        

    def _backtrack(self) -> None:
        """Returns to the parent of the current node (update plan and cur_node)."""
        this_subtask = self.degra_subtask_dict[self.cur_node["degradation"]]
        self.plan.insert(0, this_subtask)

        parent_img_path = next(
            Path(self.cur_node["img_path"]).parents[3].glob("0-img/*.png")
        )
        self.cur_node = self._img_path_to_node(parent_img_path)
        self.workflow_logger.info(
            f"Back to {self._img_nickname(self.cur_node['img_path'])}.")
        

    def _img_path_to_node(self, img_path: Path) -> dict:
        subtasks, tools = self._get_execution_path(img_path)
        node = self.work_mem["tree"]
        for subtask, tool in zip(subtasks, tools):
            node = node["children"][subtask]["tools"][tool]
        return node
    

    def reschedule(self) -> None:
        if not self.plan:
            return
        
        if not self.cur_node["children"]:
            # compromise, pick up the failed plan
            done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
            for adjusted_plan in self.work_mem["plan"]["adjusted"]:
                failed = adjusted_plan["failed"]
                failed_done, failed_planned = failed.split(" + ")
                failed_done, failed_planned = eval(failed_done), eval(failed_planned)
                if failed_done == done_subtasks:
                    self.plan = failed_planned
                    self.workflow_logger.info(f"Pick up the failed plan {failed_done} + {failed_planned}.")
                    break
            else:
                raise Exception(f"Invalid rescheduling: no failed plan found when processing {self.work_dir}.")

        elif len(self.plan) == len(self.cur_node["children"]) + 1:
            next_agenda = list(self.cur_node["children"])
            next_plan = self.schedule(next_agenda)
            top_subtask = list(set(self.plan)-set(next_agenda))[0]
            self.plan = [top_subtask] + next_plan

        else:
            done_top_subtasks = list(self.cur_node["children"])
            assert len(self.plan) - len(done_top_subtasks) > 1
            if len(done_top_subtasks) == 1:
                failed_tries_str = done_top_subtasks[0]
            else:
                failed_tries_str = 'any of ' + ', '.join(done_top_subtasks)
            reschedule_ps = prompts.reschedule_ps_prompt.format(
                failed_tries=failed_tries_str)
            self.plan = self.schedule(agenda=self.plan, ps=reschedule_ps)

            if self.plan[0] in done_top_subtasks:
                invalid_plan = self.plan.copy()
                for i, subtask in enumerate(self.plan):
                    if subtask not in done_top_subtasks:
                        self.plan[0], self.plan[i] = self.plan[i], self.plan[0]
                        break
                self.workflow_logger.warning(
                    f"Invalid rescheduling: the first subtask of {invalid_plan} "
                    f"in {done_top_subtasks}. Swapping it with {self.plan[0]}.")

        # record update
        done_subtasks, _ = self._get_execution_path(Path(self.cur_node['img_path']))
        assert set(done_subtasks+self.plan) == set(self.work_mem["plan"]["initial"]), \
            (f"Invalid adjusted plan: {done_subtasks} ∪ {self.plan} "
             f"!= {self.work_mem['plan']['initial']}.")
        self.work_mem["plan"]["adjusted"][-1]["new"] = f"{done_subtasks} + {self.plan}"
        self._dump_summary()

        self.workflow_logger.info(f"Adjusted plan: {self.plan}.")
        

    def _prepare_for_subtask(
        self, subtask: Subtask
    ) -> tuple[Path, Degradation, list[Tool]]:
        self.workflow_logger.info(
            f"Executing {subtask} on {self._img_nickname(self.cur_node['img_path'])}..."
        )

        subtask_dir = Path(self.cur_node["img_path"]).parents[1] / f"subtask-{subtask}"
        subtask_dir.mkdir()

        degradation = self.subtask_degra_dict[subtask]
        toolbox = self.executor.toolbox_router[subtask]
        random.shuffle(toolbox)

        return subtask_dir, degradation, toolbox
    
    
    def _prepare_for_subtask_face_restore(
        self, subtask: Subtask
    ) -> tuple[Path, list[Tool]]:
        self.workflow_logger.info(
            f"Executing {subtask} ..."
        )

        subtask_dir = Path(self.cur_node["img_path"]).parents[1] / f"subtask-{subtask}"
        subtask_dir.mkdir()

        # degradation = self.subtask_degra_dict[subtask]
        toolbox = self.executor.toolbox_router[subtask]
        random.shuffle(toolbox)

        return subtask_dir, toolbox
    

    def _record_tool_res(self, img_path: Path) -> None:
        tool_name = self._get_name_stem(img_path.parents[1].name)
        subtask = self._get_name_stem(img_path.parents[2].name)
        degradation = self.subtask_degra_dict[subtask]

        self.workflow_logger.info(
            f"{tool_name} is used in restoration from {degradation}, sequence: {self._img_nickname(img_path)} ."
        )

        # update working memory
        cur_children = self.cur_node["children"]
        if subtask not in cur_children:
            cur_children[subtask] = {"best_tool": None, "tools": {}}
        assert tool_name not in cur_children[subtask]["tools"]
        cur_children[subtask]["tools"][tool_name] = {
            "degradation": degradation,
            "img_path": str(img_path),
            "best_descendant": None,
            "children": {},
        }
        

    def _record_res(self) -> None:
        self.res_path = Path(self.cur_node["img_path"])
        self.workflow_logger.info(
            f"Restoration result: {self._img_nickname(self.res_path)}.")
        subtasks, tools = self._get_execution_path(self.res_path)
        self.work_mem["execution_path"]["subtasks"] = subtasks
        self.work_mem["execution_path"]["tools"] = tools
        self._dump_summary()
        shutil.copy(self.res_path, self.work_dir / "result.png")
        

    def _get_execution_path(self, img_path: Path) -> tuple[list[Subtask], list[ToolName]]:
        """Returns the execution path of the restored image (list of subtask and tools)."""
        exe_path = self._img_tree.get_execution_path(img_path)
        if not exe_path:
            return [], []
        subtasks, tools = zip(*exe_path)
        return list(subtasks), list(tools)
    

    def _prepare_dir(self, input_path: Path, output_dir: Path) -> None:
        """Sets attributes: `work_dir, img_tree_dir, log_dir, qa_path, workflow_path, summary_path`. Creates necessary directories, which will be like
        ```
        output_dir
        └── {task_id}(work_dir)
            ├── img_tree
            │   └── 0-img
            │       └── input.png
            └── logs
                ├── summary.json
                ├── workflow.log
                ├── llm_qa.md
                └── img_tree.html
        ```
        """

        task_id = f"{input_path.stem}-{strftime('%y%m%d_%H%M%S', localtime())}"
        self.work_dir = output_dir / task_id
        self.work_dir.mkdir(parents=True)

        self.img_tree_dir = self.work_dir / "img_tree"
        self.img_tree_dir.mkdir()

        self.faces_dir = self.work_dir / "faces"
        self.faces_dir.mkdir()

        self.log_dir = self.work_dir / "logs"
        self.log_dir.mkdir()
        self.qa_path = self.log_dir / "llm_qa.md"
        self.workflow_path = self.log_dir / "workflow.log"
        self.work_mem_path = self.log_dir / "summary.json"

        rqd_input_dir = self.img_tree_dir / "0-img"
        rqd_input_dir.mkdir()
        rqd_input_path = rqd_input_dir / "input.png"
        self.root_input_path = rqd_input_path
        
        # Check if input_path is a PNG image; if not, convert to PNG before copying
        if input_path.suffix.lower() != ".png":
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                img.save(rqd_input_path)
        else:
            shutil.copy(input_path, rqd_input_path)

        self._render_img_tree()
        

    def _img_nickname(self, img_path: str | Path) -> str:
        """Image name to display in log, showing the execution path."""        
        if isinstance(img_path, str):
            img_path = Path(img_path)
        subtasks, tools = self._get_execution_path(img_path)
        if not subtasks:
            return "input"
        return "-".join([f"{subtask}@{tool}" 
                         for subtask, tool in zip(subtasks, tools)])
        

    def _get_name_stem(self, name: str) -> str:
        return name[name.find("-") + 1 :]
    

    @property
    def _img_tree(self) -> ImgTree:
        return ImgTree(self.img_tree_dir, html_dir=self.log_dir)
    

    def _render_img_tree(self) -> None:
        self._img_tree.to_html()
        

    def _dump_summary(self) -> None:
        with open(self.work_mem_path, "w") as f:
            json.dump(self.work_mem, f, indent=2)
