import os
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from typing import (
    List,
    Optional,
    Set,
    Type,
)
from dflow import (
    InputArtifact,
    InputParameter,
    Inputs,
    OutputArtifact,
    OutputParameter,
    Outputs,
    Step,
    Steps,
    Workflow,
    argo_len,
    argo_range,
    argo_sequence,
    download_artifact,
    upload_artifact,
)
from dflow.python import (
    OP,
    OPIO,
    Artifact,
    OPIOSign,
    PythonOPTemplate,
    Slices,
)
from dflow.plugins.dispatcher import DispatcherExecutor


class SimplePropertyFlow(Steps):
    def __int__(
        self,
        name: str,
        make_op: Type[OP],
        lmp_run_op: Type[OP],
        vasp_run_op: Type[OP],
        abacus_run_op: Type[OP],
        post_op: Type[OP],
        make_image: str,
        run_image: str,
        post_image: str,
        run_command: str,
        calculator: str,
        executor: Optional[DispatcherExecutor] = None,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):

        self._input_parameters = {
            "flow_id": InputParameter(type=str, value=""),
            "prop_param": InputParameter(type=dict),
            "inter_param": InputParameter(type=dict),
            "do_refine": InputParameter(type=bool)
        }
        self._input_artifacts = {
            "path_to_work": InputArtifact(type=Path),
            "path_to_equi": InputArtifact(type=Path)
        }
        self._output_parameters = {}
        self._output_artifacts = {
            "output_post": OutputArtifact(type=Path)
        }

        super().__init__(
            name=name,
            inputs=Inputs(
                parameters=self._input_parameters,
                artifacts=self._input_artifacts
            ),
            outputs=Outputs(
                parameters=self._output_parameters,
                artifacts=self._output_artifacts
            ),
        )

        self._keys = ["make", "run", "post"]
        self.step_keys = {}
        key = "make"
        self.step_keys[key] = '--'.join(
            [self.inputs.parameters["flow_id"], key]
        )
        key = "run"
        self.step_keys[key] = '--'.join(
            [self.inputs.parameters["flow_id"], key + "-{{item}}"]
        )
        key = "post"
        self.step_keys[key] = '--'.join(
            [self.inputs.parameters["flow_id"], key]
        )

        self._build()

    @property
    def input_parameters(self):
        return self._input_parameters

    @property
    def input_artifacts(self):
        return self._input_artifacts

    @property
    def output_parameters(self):
        return self._output_parameters

    @property
    def output_artifacts(self):
        return self._output_artifacts

    @property
    def keys(self):
        return self._keys

    def _build(
        self,
        name: str,
        make_op: Type[OP],
        lmp_run_op: Type[OP],
        vasp_run_op: Type[OP],
        abacus_run_op: Type[OP],
        post_op: Type[OP],
        make_image: str,
        run_image: str,
        post_image: str,
        run_command: str,
        calculator: str,
        executor: Optional[DispatcherExecutor] = None,
        upload_python_packages: Optional[List[os.PathLike]] = None,
    ):
        # Step for property make
        make = Step(
            name="prop-make",
            template=PythonOPTemplate(make_op, image=make_image, command=["python3"]),
            artifacts={"path_to_work": self.inputs.artifacts["path_to_work"],
                       "path_to_equi": self.inputs.artifacts["path_to_equi"]},
            parameters={"prop_param": self.inputs.parameters["prop_param"],
                        "inter_param": self.inputs.parameters["inter_param"],
                        "do_refine": self.inputs.parameters["do_refine"]},
            key=self.step_keys["make"]
        )
        self.add(make)

        # Step for property run
        if calculator == 'vasp':
            run = PythonOPTemplate(
                vasp_run_op,
                slices=Slices(
                    "{{item}}",
                    input_parameter=["task_name"],
                    input_artifact=["task_path"],
                    output_artifact=["backward_dir"]
                ),
                python_packages=upload_python_packages,
                image=run_image
            )
            runcal = Step(
                name="PropsVASP-Cal",
                template=run,
                parameters={
                    "run_image_config": {"command": run_command},
                    "task_name": make.outputs.parameters["task_names"],
                    "backward_list": ["INCAR", "POSCAR", "OUTCAR", "CONTCAR"]
                },
                artifacts={
                    "task_path": make.outputs.artifacts["task_paths"]
                },
                with_param=argo_range(argo_len(make.outputs.parameters["task_names"])),
                key=self.step_keys["run"] + '-vasp',
                executor=executor,
            )
        elif calculator == 'abacus':
            run = PythonOPTemplate(
                abacus_run_op,
                slices=Slices(
                    "{{item}}",
                    input_parameter=["task_name"],
                    input_artifact=["task_path"],
                    output_artifact=["backward_dir"]
                ),
                python_packages=upload_python_packages,
                image=run_image
            )
            runcal = Step(
                name="PropsABACUS-Cal",
                template=run,
                parameters={
                    "run_image_config": {"command": run_command},
                    "task_name": make.outputs.parameters["task_names"],
                    "backward_list": ["OUT.ABACUS", "log"],
                    "log_name": "log"
                },
                artifacts={
                    "task_path": make.outputs.artifacts["task_paths"],
                    "optional_artifact": upload_artifact({"pp_orb": "./"})
                },
                with_param=argo_range(argo_len(make.outputs.parameters["task_names"])),
                key=self.step_keys["run"] + '-abacus',
                executor=executor,
            )
        elif calculator == 'lammps':
            run = PythonOPTemplate(
                lmp_run_op,
                slices=Slices(
                    "{{item}}",
                    input_artifact=["input_lammps"],
                    output_artifact=["output_lammps"]
                ),
                image=run_image,
                command=["python3"]
            )
            runcal = Step(
                name="PropsLAMMPS-Cal",
                template=run,
                artifacts={"input_lammps": make.outputs.artifacts["task_paths"]},
                parameters={"run_command": run_command},
                with_param=argo_range(make.outputs.parameters["njobs"]),
                key=self.step_keys["run"] + '-lammps',
                executor=executor,
            )
        else:
            raise RuntimeError(f'Incorrect calculator type to initiate step: {calculator}')
        self.add(runcal)

        # Step for property post
        if calculator in ['vasp', 'abacus']:
            post = Step(
                name="Props-post",
                template=PythonOPTemplate(
                    post_op,
                    image=post_image,
                    command=["python3"]
                ),
                artifacts={"input_post": runcal.outputs.artifacts["backward_dir"],
                           "input_all": make.outputs.artifacts["output_work_path"]},
                parameters={"prop_param": self.inputs.parameters["prop_param"],
                            "inter_param": self.inputs.parameters["inter_param"],
                            "task_names": make.outputs.parameters["task_names"]},
                key=self.step_keys["post"]
            )
        elif calculator == 'lammps':
            post = Step(
                name="Props-post",
                template=PythonOPTemplate(
                    post_op,
                    image=post_image,
                    command=["python3"]
                ),
                artifacts={"input_post": runcal.outputs.artifacts["output_lammps"],
                           "input_all": make.outputs.artifacts["output_work_path"]},
                parameters={"prop_param": self.inputs.parameters["prop_param"],
                            "inter_param": self.inputs.parameters["inter_param"],
                            "task_names": make.outputs.parameters["task_names"]},
                key=self.step_keys["post"]
            )
            self.add(post)

            self.outputs.artifacts["output_post"]._from = post.outputs.artifacts["output_post"]




