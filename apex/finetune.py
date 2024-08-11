import os
import os.path
import glob
import logging
from typing import List
from monty.serialization import loadfn, dumpfn
from pathlib import Path
from abc import ABC, abstractmethod

from dpdata.periodic_table import Element

from apex.submit import submit_workflow
from apex.step import do_step
from apex.core.constants import PERIOD_ELEMENTS_BY_SYMBOL

from apex.utils import (
    judge_flow,
    load_config_file,
    handle_prop_suffix,
)


class ConcurrentLearningFramework(ABC):
    @abstractmethod
    def __init__(self, config_file: os.PathLike):
        pass

    @abstractmethod
    def prepare(self, target_confs: dict[os.PathLike: List[os.PathLike]]):
        pass

    @abstractmethod
    def invoke(self):
        pass


class DPGen2(ConcurrentLearningFramework):
    def __init__(self, config_file: os.PathLike):
        self._config = loadfn(str(config_file))
        self.config_file = "dpgen2_input.json"
        try:
            self.trj_freq = self._config["explore"]["stages"][0][0]["trj_freq"]
        except [KeyError, IndexError]:
            self.trj_freq = 10
        try:
            self.n_sample = self._config["explore"]["stages"][0][0]["n_sample"]
        except [KeyError, IndexError]:
            self.n_sample = 20

    @property
    def config(self):
        return self._config

    @staticmethod
    def modify_inlammps(inlammps: os.PathLike, num_elements: int = 103):
        with open(inlammps, 'r') as f:
            lines = f.readlines()
        mass_line_indices = []
        for idx, line in enumerate(lines):
            # if line starts with "dump", replace the word after "dump" by "dpgen2_dump
            if line.startswith("dump"):
                new_line = line.split()
                new_line[1] = "dpgen_dump"
                lines[idx] = " ".join(new_line) + "\n"
            if line.startswith("read_data"):
                mass_line_idx = idx + 1
            # if line starts with "mass", delete the line, and record the line index
            if line.startswith("mass"):
                mass_line_indices.append(idx)
            if line.startswith("pair_coeff"):
                elements = " ".join(PERIOD_ELEMENTS_BY_SYMBOL[:num_elements])
                lines[idx] = f"pair_coeff * * {elements}\n"
        # Delete the mass lines after the loop
        for idx in reversed(mass_line_indices):
            del lines[idx]
        # insert new mass lines at mass_line_idx
        new_mass_line = ""
        for ii in range(num_elements):
            new_mass_line += "mass            %d %.3f\n" % (ii + 1, Element(PERIOD_ELEMENTS_BY_SYMBOL[ii]).mass)
        lines.insert(mass_line_idx, new_mass_line)
        with open(inlammps, 'w') as f:
            f.writelines(lines)

    def prepare(self, target_confs: dict[os.PathLike: List[os.PathLike]]):
        self._config["explore"]["configuration_prefix"] = None
        sys_idx = 0
        configurations_list = []
        stages_list = []
        for inlammps, poscars in target_confs.items():
            self.modify_inlammps(inlammps)
            configurations_list.append({
                "type": "file",
                "files": poscars,
                "fmt": "vasp/poscar"
            })
            stages_list.append({
                "type": "lmp-template",
                "lmp": inlammps,
                "trj_freq": self.trj_freq,
                "sys_idx": [sys_idx],
                "n_sample": self.n_sample
            })
            sys_idx += 1
        self._config["explore"]["configurations"] = configurations_list
        self._config["explore"]["stages"] = [stages_list]
        dumpfn(self._config, self.config_file, indent=2)

    def invoke(self):
        os.system(f"dpgen2 submit {self.config_file}")


def invoke_external_concurrent_learning_framework(
    method: str = 'dpgen2',
    external_config_template: os.PathLike = None,
    target_confs: dict[os.PathLike: List[os.PathLike]] = None,
    prepare_only: bool = False,
):
    if method == "dpgen2":
        framework = DPGen2(external_config_template)
    else:
        raise ValueError(f'Unsupported method: {method}')
    print('Writing input files...')
    framework.prepare(target_confs)
    if not prepare_only:
        print(f'Invoking external concurrent learning framework {method}...')
        framework.invoke()


def check_relaxation(
    param: dict,
    work_dir: os.PathLike,
):
    wd = Path(work_dir)
    confs = param["structures"]
    for ii in confs:
        for jj in wd.glob(ii):
            relax_result = jj / 'relaxation/relax_task/result.json'
            if not relax_result.exists():
                logging.debug(f'No relaxation result found for {jj}')
                return False
    return True


def check_props_make(
    param: dict,
    work_dir: os.PathLike,
):
    wd = Path(work_dir)
    confs = param["structures"]
    properties = param["properties"]
    conf_dirs = []
    path_to_prop_list = []
    do_make = False
    for conf in confs:
        conf_dirs.extend(glob.glob(str(wd / conf)))
    conf_dirs = list(set(conf_dirs))
    conf_dirs.sort()
    element_set = set()
    for ii in conf_dirs:
        poscar = os.path.join(ii, 'POSCAR')
        with open(poscar, 'r') as f:
            for jj in f.readlines()[5].split():
                element_set.add(jj)
        for jj in properties:
            do_refine, suffix = handle_prop_suffix(jj)
            if not suffix:
                continue
            property_type = jj["type"]
            path_to_prop = os.path.join(ii, property_type + "_" + suffix)
            path_to_prop_list.append(path_to_prop)
            if not os.path.exists(path_to_prop):
                do_make = True
    return do_make, path_to_prop_list, element_set


def finetune(
    parameter_dicts: List[dict],
    config_dict: dict,
    external_config_template: os.PathLike,
    method: str,
    work_dirs: List[os.PathLike],
    is_debug=False,
    prepare_only=False,
):
    assert len(work_dirs) == 1, 'Only one work directory is supported for finetune mode.'
    work_dir = work_dirs[0]
    cwd = os.getcwd()
    os.chdir(work_dir)
    _, _, _, relax_param, props_param = judge_flow(parameter_dicts, specify='joint')

    print('===>>> STEP 1: Submit Relaxation APEX Workflow')
    if check_relaxation(relax_param, work_dir):
        print('Relaxation results found. Skip the STEP 1.')
    else:
        submit_workflow(
            parameter_dicts=parameter_dicts,
            config_dict=config_dict,
            work_dirs=work_dirs,
            indicated_flow_type='relax',
            is_debug=is_debug,
        )

    print('===>>> STEP 2: Generate Target Property Configurations to Finetune')
    do_make, path_to_prop_list, element_set = check_props_make(props_param, work_dir)
    if do_make:
        print('Making property configurations...')
        props_param["interaction"]["type"] = "deepmd"
        props_param["interaction"]["model"] = "tmp_pseudo_pot.pb"
        all_element = list(element_set)
        type_map = {all_element[i]: i for i in range(len(all_element))}
        props_param["interaction"]["type_map"] = type_map
        tmp_pb = Path(work_dir) / "tmp_pseudo_pot.pb"
        tmp_pb.touch(exist_ok=True)
        do_step(
            param_dict=props_param,
            step='make_props',
        )
        os.remove(tmp_pb)
    else:
        print('Property configurations found. Collect them...')
    target_confs = {}
    for ii in path_to_prop_list:
        poscars = glob.glob(os.path.join(ii, 'task.000*/POSCAR'))
        inlammps = glob.glob(os.path.join(ii, 'in.lammps'))[0]
        target_confs[inlammps] = poscars

    print('===>>> STEP 3: Invoke External Concurrent Learning Framework')
    invoke_external_concurrent_learning_framework(
        method=method,
        external_config_template=external_config_template,
        target_confs=target_confs,
        prepare_only=prepare_only,
    )
    os.chdir(cwd)


def finetune_from_args(
    parameters,
    config_file: os.PathLike,
    external_config_template: os.PathLike,
    work_dirs,
    method: str,
    is_debug=False,
    prepare_only=False,
):
    print('-------Finetune Mode-------')
    finetune(
        parameter_dicts=[loadfn(jj) for jj in parameters],
        config_dict=load_config_file(config_file),
        work_dirs=work_dirs,
        external_config_template=external_config_template,
        method=method,
        is_debug=is_debug,
        prepare_only=prepare_only,
    )
    print('Completed!')
