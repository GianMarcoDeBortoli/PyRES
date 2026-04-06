# ==================================================================
# ============================ IMPORTS =============================
from collections import OrderedDict
import os
import time
# PyTorch
import torch
import torch.nn as nn
# FLAMO
from flamo import dsp, system
from flamo.functional import db2mag, mag2db, get_magnitude, get_eigenvalues
# PyRES
from PyRES.physical_room import _PhRoom
from PyRES.virtual_room import _VrRoom
from PyRES.utils import expand_to_dimension


# ==================================================================
# ================ REVERBERATION ENHANCEMENT SYSTEM ================

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
            physical_room: _PhRoom,
            virtual_room: _VrRoom,
        ):
        r"""
        Initializes the Reverberation Enhancement System (RES).

            **Args**:
                - physical_room (PhRoom): physical room.
                - virtual_room (VrRoom): virtual room.

            **Attributes**:
                - fs (torch.Tensor): sampling frequency [Hz].
                - nfft (int): FFT size.
                - alias_decay_db (float): anti-time-aliasing decay [dB].
                - n_S (int): number of stage sources.
                - n_M (int): number of system microphones.
                - n_L (int): number of system loudspeakers.
                - n_A (int): number of audience positions.
                - __H (PhRoom): physical room.
                - __v_ML (VrRoom): virtual room.
                - __G (dsp.parallelGain): system gain.
        """
        object.__init__(self)

        # Processing parameters
        self.fs, self.nfft, self.alias_decay_db = self.__check_param_compatibility(physical_room, virtual_room)

        # Number of emitters and receivers
        self.transducer_number = self.__check_io_compatibility(physical_room, virtual_room)

        # Physical room
        self.phroom = physical_room

        # Virtual room
        self.vrroom = virtual_room

        # System gain
        self.G = dsp.parallelGain(size=(self.transducer_number['lds'],), nfft=self.nfft, alias_decay_db=self.alias_decay_db)

        # Apply safe margin of 2 dB
        gbi_init = self.compute_GBI()
        self.set_G(db2mag(mag2db(gbi_init) - 3))
    
    # ==================================================================================
    # ================================ CHECK METHODS ===================================

    def __check_param_compatibility(self, physical_room: _PhRoom, virtual_room: _VrRoom) -> torch.Tensor:

        assert(physical_room.fs == virtual_room.fs), "Sampling frequency must be the same in physical and virtual rooms."
        assert(physical_room.nfft == virtual_room.nfft), "Number of frequency bins must be the same in physical and virtual rooms."
        assert(physical_room.alias_decay_db == virtual_room.alias_decay_db), "Anti-time-aliasing decay must be the same in physical and virtual rooms."

        return physical_room.fs, physical_room.nfft, physical_room.alias_decay_db

    def __check_io_compatibility(self, physical_room: _PhRoom, virtual_room: _VrRoom) -> tuple[int, int, int, int]:
        
        assert(physical_room.transducer_number['mcs'] == virtual_room.n_M), "Number of microphones must be the same in physical and virtual rooms."
        assert(physical_room.transducer_number['lds'] == virtual_room.n_L), "Number of loudspeakers must be the same in physical and virtual rooms."

        return physical_room.transducer_number

    # ==================================================================================
    # ============================ PHYSICAL ROOM METHODS ===============================

    def get_h_SA(self) -> nn.Module:
        f"""
        Returns the room impulse responses between stage sources and audience positions.

            **Returns**:
                - nn.Module: Stage-to-Audience RIRs
        """
        return self.phroom.get_stg_to_aud()
    
    def get_h_SM(self) -> nn.Module:
        f"""
        Returns the room impulse responses between stage sources and system microphones.

            **Returns**:
                - nn.Module: Stage-to-Microphones RIRs
        """
        return self.phroom.get_stg_to_mcs()
    
    def get_h_LA(self) -> nn.Module:
        f"""
        Returns the room impulse responses between system loudspeakers and audience positions.

            **Returns**:
                - nn.Module: Loudspeaker-to-Audience RIRs
        """
        return self.phroom.get_lds_to_aud()

    def get_h_LM(self) -> nn.Module:
        f"""
        Returns the room impulse responses between system loudspeakers and system microphones.

            **Returns**:
                - nn.Module: Loudspeaker-to-Microphones RIRs
        """
        return self.phroom.get_lds_to_mcs()
    
    # ==================================================================================
    # ============================= VIRTUAL ROOM METHODS ===============================

    def get_v_ML(self) -> nn.Module:
        f"""
        Returns the virtual room.

            **Returns**:
                - nn.Module: virtual room.
        """
        return self.vrroom.get_v_ML()
    
    def get_v_ML_responses(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time and frequency responses of the virtual room.
        
            **Returns**:
                - torch.Tensor [samples, n_L, n_M]: time responses
                - torch.Tensor [nfft, n_L, n_M]: frequency responses
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
        Returns the system-gain value in linear scale.

            **Returns**:
                - torch.Tensor: system gain value (linear scale).
        """
        return self.G

    def set_G(self, g: float) -> None:
        r"""
        Sets the system gain value to a value in linear scale.

            **Args**:
                - g (float): new system gain value (linear scale).
        """
        assert isinstance(g, torch.FloatTensor), "G must be a torch.FloatTensor."
        self.G.assign_value(g*torch.ones(self.transducer_number['lds']))

    def compute_GBI(self, criterion: str='eigenvalue_magnitude') -> torch.Tensor:
        r"""
        Returns the system Gain Before Instability (GBI) value in linear scale.

            **Args**:
                - criterion (str, optional): criterion to compute the GBI. Defaults to 'eigenvalue_magnitude'.

            **Returns**:
                - torch.Tensor: GBI value (linear scale).
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
    
    def set_G_to_GBI(self, gbi_estimation_criterion: str='eigenvalue_magnitude') -> None:
        r"""
        Sets the system gain to match the current system GBI.

            **Args**:
                - gbi_estimation_criterion (str, optional): criterion to compute the GBI. Defaults to 'eigenvalue_magnitude'.
        """
        # Compute the current gain before instability
        gbi = self.compute_GBI(criterion=gbi_estimation_criterion)

        # Apply the current gain before instability to the system gain module
        self.set_G(gbi)

    # ==================================================================================
    # ================================= FEEDBACK LOOP ==================================

    def open_loop(self, state: str='mcs')-> system.Series:
        r"""
        Generates the system open loop.

            **Args**:
                - state (str, optional): state variable defining the point of start and end of the open-loop. Defaults to 'mcs'. Possible values are 'mcs' and 'lds', corresponding microphones and loudspeakers, respectively.

            **Returns**:
                - system.Series: Series object instance implementing the system open loop.
        """
        if state == 'mcs':
            modules = OrderedDict([
                ('V_ML', self.get_v_ML()),
                ('G', self.get_G()),
                ('H_LM', self.get_h_LM())
            ])
        elif state == 'lds':
            modules = OrderedDict([
                ('H_LM', self.get_h_LM()),
                ('V_ML', self.get_v_ML()),
                ('G', self.get_G())
            ])

        return system.Series(modules)
    
    def open_loop_responses(self, state: str='mcs') -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the time- and frequency-response matrices of the open-loop.

            **Args**:
                - state (str, optional): state variable defining the point of start and end of the open-loop. Defaults to 'mcs'. Possible values are 'mcs' and 'lds', corresponding microphones and loudspeakers, respectively.

            **Returns**:
                - torch.Tensor: time responses [samples, n_M, n_M].
                - torch.Tensor: frequency responses [nfft, n_M, n_M].
        """

        # Generate open loop
        if state == 'mcs':
            open_loop = system.Shell(
                core=self.open_loop(state=state)
            )
        elif state == 'lds':
            open_loop = system.Shell(
                core=self.open_loop(state=state)
            )
        
        with torch.no_grad():
            # Compute open-loop time and frequency responses
            open_loop_irs = open_loop.get_time_response(fs=self.fs, identity=True).squeeze() # TODO: Check if squeeze is needed
            open_loop_fr = open_loop.get_freq_response(fs=self.fs, identity=True).squeeze()

        # Expand to the explicit dimension
        open_loop_irs = expand_to_dimension(array=open_loop_irs, dim=2)
        open_loop_fr = expand_to_dimension(array=open_loop_fr, dim=2)

        return open_loop_irs, open_loop_fr

    def open_loop_eigenvalues(self) -> torch.Tensor:
        r"""
        Computes the eigenvalues of the system open loop.

            **Returns**:
                - torch.Tensor: open-loop eigenvalues [nfft, n_M, n_M].
        """

        # Generate open-loop frequency responses
        _, fr_matrix = self.open_loop_responses()
        with torch.no_grad():
            # Compute eigenvalues
            evs = get_eigenvalues(fr_matrix).squeeze()

        # Expand to the explicit dimension
        evs = expand_to_dimension(array=evs, dim=2)

        return evs
    
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
                - torch.Tensor: time responses [samples, n_M, n_M].
                - torch.Tensor: frequency responses [nfft, n_M, n_M].
        """

        # Generate closed loop
        closed_loop = system.Shell(
            core=self.closed_loop()
        )
        
        with torch.no_grad():
            # Compute closed-loop time and frequency responses
            closed_loop_irs = closed_loop.get_time_response(fs=self.fs, identity=True).squeeze()
            closed_loop_fr = closed_loop.get_freq_response(fs=self.fs, identity=True).squeeze()

        # Expand to the explicit dimension
        closed_loop_irs = expand_to_dimension(array=closed_loop_irs, dim=2)
        closed_loop_fr = expand_to_dimension(array=closed_loop_fr, dim=2)

        return closed_loop_irs, closed_loop_fr
    
    # ==================================================================================
    # =============================== SYSTEM SIMULATION ================================

    def __system_paths(self) -> tuple[system.Shell, system.Shell]:
        r"""
        Creates the full system's Natural and Electroacoustic paths.

            **Returns**:
                - Shell: Shell object instance implementing the natural path of the RES.
                - Shell: Shell object instance implementing the electroacoustic path of the RES.
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
    
    def system_simulation(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Simulates the full system producing the system impulse responses from the stage emitters to the audience receivers.

            **Returns**:
                - torch.Tensor: natural room impulse responses of the physical room [samples, n_A, n_S].
                - torch.Tensor: electroacoustic impulse responses of the RES [samples, n_A, n_S].
                - torch.Tensor: full system impulse responses [samples, n_A, n_S].
        """
        # Generate the paths
        nat_path, ea_path = self.__system_paths()
        
        with torch.no_grad():
            # Compute system response
            y_nat = nat_path.get_time_response().squeeze() # TODO: Check if squeeze is needed
            y_ea = ea_path.get_time_response().squeeze()

        # Expand to the explicit dimension
        y_nat = expand_to_dimension(array=y_nat, dim=2)
        y_ea = expand_to_dimension(array=y_ea, dim=2)

        return y_nat, y_ea, y_nat + y_ea
    
    # ==================================================================================
    # ================================= SYSTEM STATE ===================================

    def get_v_ML_state(self) -> dict:
        r"""
        Returns the system current state.

            **Returns**:
                - dict: model's state.
        """
        return self.get_v_ML().state_dict()
    
    def set_v_ML_state(self, state: dict) -> None:
        r"""
        Sets the system current state.

            **Args**:
                - state (dict): new state.
        """
        self.get_v_ML().load_state_dict(state)

    def save_state_to(self, directory: str) -> None:
        r"""
        Saves the system current state.

            **Args**:
                - directory (str): path to save the state.
        """
        directory = directory.rstrip('/')
        state = self.get_v_ML_state()
        torch.save(state, os.path.join(directory, time.strftime("%Y-%m-%d_%H.%M.%S.pt")))
