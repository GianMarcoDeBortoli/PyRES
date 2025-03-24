# ==================================================================
# ============================ IMPORTS =============================
# Miscellanous
from collections import OrderedDict
import os
import time
# Torch
import torch
import torch.nn as nn
# Flamo
from flamo import dsp, system
from flamo.functional import db2mag, mag2db, get_magnitude, get_eigenvalues
# PyRES
from pyRES.physical_room import PhRoom
from pyRES.virtual_room import VrRoom


# ==================================================================
# ================ REVERBERATION ENHANCEMENT SYSTEM ================]

class RES(object):
    r"""
    Template for the Reverberation Enhancement System (RES) model.

        **Stores**:
            - system parameters
            - physical-room data
            - virtual-room data

        **Functionalities**:
            - system gain control
            - virtual-room time- and frequency-response matrices computation
            - system open loop time- and frequency-response matrices and eigenvalues computation
            - system closed loop time- and frequency-response matrices computation
            - full system simulation
            - optimization routine control for virtual-room parameter learning
            - system state management
    """
    def __init__(
            self,
            physical_room: PhRoom,
            virtual_room: VrRoom,
        ):
        r"""
        Initializes the Reverberation Enhancement System (RES).

            **Args**:
                - physical_room (PhRoom): physical room.
                - virtual_room (VrRoom): virtual room. Defaults to None.
        """
        object.__init__(self)

        # Processing parameters
        self.fs, self.nfft, self.alias_decay_db = self.__check_param_compatibility(physical_room, virtual_room)

        # Number of emitters and receivers
        self.n_S, self.n_A, self.n_M, self.n_L = self.__check_io_compatibility(physical_room, virtual_room)

        # Physical room
        self.__H = physical_room

        # Virtual room
        self.__v_ML = virtual_room

        # System gain
        self.__G = dsp.parallelGain(size=(self.n_L,), nfft=self.nfft, alias_decay_db=self.alias_decay_db)

        # Apply safe margin of 2 dB
        gbi_init = self.compute_GBI()
        self.set_G(db2mag(mag2db(gbi_init) - 2))

        # Optimization routine
        # self.__opt = system.Shell(
        #     core=self.open_loop(),
        #     input_layer=system.Series(
        #         dsp.Transform(lambda x: x.diag_embed()),
        #         dsp.FFT(self.nfft)
        #     )
        # )

    # ==================================================================================
    # ================================== FORWARD PATH ==================================

    # TODO: cancel this
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     r"""
    #     Computes one iteration of the optimization routine.

    #         **Args**:
    #             x (torch.Tensor): input signal.

    #         **Returns**:
    #             torch.Tensor: output of the optimization routine.
    #     """
    #     return self.__opt(x)
    
    # ==================================================================================
    # ================================ CHECK METHODS ===================================

    def __check_param_compatibility(self, physical_room: PhRoom, virtual_room: VrRoom) -> torch.Tensor:

        assert(physical_room.fs == virtual_room.fs), "Sampling frequency must be the same in physical and virtual rooms."
        assert(physical_room.nfft == virtual_room.nfft), "Number of frequency bins must be the same in physical and virtual rooms."
        assert(physical_room.alias_decay_db == virtual_room.alias_decay_db), "Anti-time-aliasing decay must be the same in physical and virtual rooms."

        return physical_room.fs, physical_room.nfft, physical_room.alias_decay_db

    def __check_io_compatibility(self, physical_room: PhRoom, virtual_room: VrRoom) -> tuple[int, int, int, int]:
        
        assert(physical_room.n_M == virtual_room.n_M), "Number of microphones must be the same in physical and virtual rooms."
        assert(physical_room.n_L == virtual_room.n_L), "Number of loudspeakers must be the same in physical and virtual rooms."

        return physical_room.n_S, physical_room.n_A, physical_room.n_M, physical_room.n_L

    # ==================================================================================
    # ============================ PHYSICAL ROOM METHODS ===============================

    def get_h_SA(self) -> nn.Module:
        f"""
        Returns the physical-room module from sound sources to audience positions.

            **Returns**:
                nn.Module: sound-source-to-audience-positions
        """
        return self.__H.get_stg_to_aud()
    
    def get_h_SM(self) -> nn.Module:
        f"""
        Returns the physical-room module from sound sources to microphones.

            **Returns**:
                nn.Module: sound-source-to-microphones
        """
        return self.__H.get_stg_to_mcs()

    def get_h_LM(self) -> nn.Module:
        f"""
        Returns the physical-room module from loudspeakers to microphones.

            **Returns**:
                nn.Module: loudspeaker-to-microphone impulse-response module.
        """
        return self.__H.get_lds_to_mcs()
    
    def get_h_LA(self) -> nn.Module:
        f"""
        Returns the physical-room module from loudspeakers to audience positions.

            **Returns**:
                nn.Module: loudspeaker-to-audience-positions
        """
        return self.__H.get_lds_to_aud()
    
    # ==================================================================================
    # ============================= VIRTUAL ROOM METHODS ===============================

    def get_v_ML(self) -> nn.Module:
        f"""
        Returns the virtual room.

            **Returns**:
                nn.Module: virtual room.
        """
        return self.__v_ML
    
    def get_v_ML_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time and frequency responses of the virtual room.
        
            **Returns**:
                tuple[torch.Tensor, torch.Tensor]: time and frequency responses
        """

        # Generate virtual room
        v_ml = system.Shell(
            core=self.get_v_ML()
        )
        with torch.no_grad():
            # Get the virtual room time and frequency responses
            v_ml_ir = v_ml.get_time_response(fs=self.fs, identity=True)
            v_ml_fr = v_ml.get_freq_response(fs=self.fs, identity=True)

        return v_ml_ir.squeeze(), v_ml_fr.squeeze()
    
    # ==================================================================================
    # ============================== SYSTEM GAIN METHODS ===============================
    
    def get_G(self) -> nn.Module:
        r"""
        Returns the system gain value in linear scale.

            **Returns**:
                torch.Tensor: system gain value (linear scale).
        """
        return self.__G

    def set_G(self, g: float) -> None:
        r"""
        Sets the system gain value to a value in linear scale.

            **Args**:
                g (float): new system gain value (linear scale).
        """
        assert isinstance(g, torch.FloatTensor), "G must be a torch.FloatTensor."
        self.__G.assign_value(g*torch.ones(self.n_L))

    def compute_GBI(self, criterion: str='eigenvalue_magnitude') -> torch.Tensor:
        r"""
        Returns the system Gain Before Instability (GBI) value in linear scale.

            **Args**:
                - criterion (str, optional): criterion to compute the GBI. Defaults to 'eigenvalue_magnitude'.

            **Returns**:
                torch.Tensor: GBI value (linear scale).
        """
        match criterion:

            case 'eigenvalue_magnitude':
                with torch.no_grad():
                    # Save current G value
                    g_current = self.get_G().param.data[0].clone().detach()

                    # Set G to 1
                    self.set_G(torch.tensor(1.00))

                    # Compute the gain before instability
                    maximum_eigenvalue = torch.max(get_magnitude(self.open_loop_eigenvalues()))
                    gbi = torch.reciprocal(maximum_eigenvalue)

                    # Restore G value
                    self.set_G(g_current)

            case 'eigenvalue_real_part':
                with torch.no_grad():
                    # Save current G value
                    g_current = self.get_G().param.data[0].clone().detach()

                    # Set G to 1
                    self.set_G(torch.tensor(1.00))

                    # Compute the gain before instability
                    maximum_eigenvalue = torch.max(torch.real(self.open_loop_eigenvalues()))
                    gbi = torch.reciprocal(maximum_eigenvalue)

                    # Restore G value
                    self.set_G(g_current)

            case _:
                raise ValueError(f"Criterion '{criterion}' not recognized.")

        return gbi
    
    def set_G_to_GBI(self) -> None:
        r"""
        Sets the system gain to match the current system GBI.
        """
        # Compute the current gain before instability
        gbi = self.compute_GBI()

        # Apply the current gain before instability to the system gain module
        self.set_G(gbi)

    # ==================================================================================
    # ================================= FEEDBACK LOOP ==================================

    def open_loop(self)-> system.Series:
        r"""
        Generates the system open loop.

            **Returns**:
                system.Series: Series object instance implementing the system open loop.
        """
        modules = OrderedDict([
            ('V_ML', self.get_v_ML()),
            ('G', self.get_G()),
            ('H_LM', self.get_h_LM())
        ])

        return system.Series(modules)
    
    def open_loop_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time- and frequency-response matrices of the open-loop.

            **Returns**:
                tuple[torch.Tensor, torch.Tensor]: time and frequency responses.
        """

        # Generate open loop
        open_loop = system.Shell(
            core=self.open_loop()
        )
        
        with torch.no_grad():
            # Compute open-loop time and frequency responses
            open_loop_irs = open_loop.get_time_response(fs=self.fs, identity=True)
            open_loop_fr = open_loop.get_freq_response(fs=self.fs, identity=True)

        return open_loop_irs.squeeze(), open_loop_fr.squeeze()

    def open_loop_eigenvalues(self) -> torch.Tensor:
        r"""
        Computes the eigenvalues of the system open loop.

            **Returns**:
                torch.Tensor: open-loop eigenvalues.
        """

        # Generate open-loop frequency responses
        _, fr_matrix = self.open_loop_responses()
        with torch.no_grad():
            # Compute eigenvalues
            evs = get_eigenvalues(fr_matrix)

        return evs.squeeze()
    
    def closed_loop(self) -> system.Recursion:
        r"""
        Generates a Recursion object instance representing the closed-loop system.

            **Returns**:
                system.Recurion: Recursion object instance implementing the closed-loop.
        """
        feedforward = system.Series(self.get_v_ML(), self.get_G())
        feedback = self.get_h_LM()
        return system.Recursion(fF=feedforward, fB=feedback)
    
    def closed_loop_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time- and frequency-response matrices of the closed-loop.

            **Returns**:
                tuple[torch.Tensor, torch.Tensor]: time and frequency responses.
        """

        # Generate closed loop
        closed_loop = system.Shell(
            core=self.closed_loop()
        )
        
        with torch.no_grad():
            # Compute closed-loop time and frequency responses
            closed_loop_irs = closed_loop.get_time_response(fs=self.fs, identity=True)
            closed_loop_fr = closed_loop.get_freq_response(fs=self.fs, identity=True)

        return closed_loop_irs.squeeze(), closed_loop_fr.squeeze()
    
    def closed_loop_eigenvalues(self) -> torch.Tensor:
        r"""
        Computes the eigenvalues of the system closed loop.

            **Returns**:
                torch.Tensor: closed-loop eigenvalues.
        """

        # Generate closed-loop frequency responses
        _, fr_matrix = self.closed_loop_responses()
        with torch.no_grad():
            # Compute eigenvalues
            evs = get_eigenvalues(fr_matrix)

        return evs.squeeze()
    
    # ==================================================================================
    # =============================== SYSTEM SIMULATION ================================

    def __system_paths(self) -> tuple[system.Shell, system.Shell]:
        r"""
        Creates the full system's Natural and Electroacoustic paths.

            **Returns**:
                tuple[Shell, Shell]: Shell object instances implementing the natural and the electroacoustic paths of the RES.
        """
        # Build closed feedback loop
        closed_loop = self.closed_loop()
        
        # Build the electroacoustic path
        ea_components = system.Series(OrderedDict([
            ('H_SM', self.get_h_SM()),
            ('FeedbackLoop', closed_loop),
            ('H_LA', self.get_h_LA())
        ]))
        ea_path = system.Shell(core=ea_components, input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))
        
        # Build the natural path
        nat_path = system.Shell(core=self.get_h_SA(), input_layer=dsp.FFT(self.nfft), output_layer=dsp.iFFT(self.nfft))

        return nat_path, ea_path
    
    def system_simulation(self) -> torch.Tensor:
        r"""
        Simulates the full system producing the system impulse responses from the stage emitters to the audience receivers.

            **Returns**:
                torch.Tensor: system impulse response.
        """
        # Generate the paths
        nat_path, ea_path = self.__system_paths()
        
        with torch.no_grad():
            # Compute system response
            y = nat_path.get_time_response() + ea_path.get_time_response()

        return y.squeeze()
    
    # ==================================================================================
    # ============================= OPTIMIZATION ROUTINE ===============================

    # def __set_opt_inputLayer(self, layer: nn.Module) -> None:
    #     r"""
    #     Sets the input layer of the optimization routine.

    #         **Args**:
    #             layer (nn.Module): input layer.
    #     """
    #     self.__opt.set_inputLayer(layer)

    # def __set_opt_outputLayer(self, layer: nn.Module) -> None:
    #     r"""
    #     Sets the output layer of the optimization routine.

    #         **Args**:
    #             layer (nn.Module): output layer.
    #     """
    #     self.__opt.set_outputLayer(layer)

    # def set_optimization_routine(self, to_optimize: str, input_layer: nn.Module=None, output_layer: nn.Module=None) -> None:
    #     r"""
    #     Sets the optimization routine.

    #         **Args**:
    #             - to_optimize (str): optimization routine.
    #             - input_layer (nn.Module, optional): input layer. Defaults to None.
    #             - output_layer (nn.Module, optional): output layer. Defaults to None.
    #     """
    #     match to_optimize:
    #         case 'open_loop':
    #             self.__set_opt_inputLayer(system.Series(
    #                 dsp.Transform(lambda x: x.diag_embed()),
    #                 dsp.FFT(self.nfft))
    #             )
    #             self.__opt = system.Shell(core=self.open_loop())
    #         case 'closed_loop':
    #             self.__set_opt_inputLayer(system.Series(
    #                 dsp.Transform(lambda x: x.diag_embed()),
    #                 dsp.FFT(self.nfft))
    #             )
    #             self.__opt = system.Shell(core=self.closed_loop())
    #         case _:
    #             raise ValueError(f"Optimization routine '{to_optimize}' not recognized.")
            
    #     if input_layer is not None:
    #         self.__set_opt_inputLayer(input_layer)
    #     if output_layer is not None:
    #         self.__set_opt_outputLayer(output_layer)
    
    # ==================================================================================
    # ================================= SYSTEM STATE ===================================

    def get_v_ML_state(self) -> dict:
        r"""
        Returns the system current state.

            **Returns**:
                dict: model's state.
        """
        return self.get_v_ML().state_dict()
    
    def set_v_ML_state(self, state: dict) -> None:
        r"""
        Sets the system current state.

            **Args**:
                state (dict): new state.
        """
        self.get_v_ML().load_state_dict(state)

    def save_state_to(self, directory: str) -> None:
        r"""
        Saves the system current state.

            **Args**:
                directory (str): path to save the state.
        """
        directory = directory.rstrip('/')
        state = self.get_v_ML_state()
        torch.save(state, os.path.join(directory, time.strftime("%Y-%m-%d_%H.%M.%S.pt")))