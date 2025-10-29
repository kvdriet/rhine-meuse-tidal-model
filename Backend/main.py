from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import json

# Import your model class
from network_model_RM import Network_model_RM

app = FastAPI(title="Rhine-Meuse Tidal Model API")

# Enable CORS for GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # Change this to your GitHub Pages URL after deployment
        # "https://yourusername.github.io",
        # "http://localhost:8080"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelParameters(BaseModel):
    """Input parameters for the tidal model"""
    friction_level: str  # 'very_low', 'low', 'baseline', 'high', 'very_high'
    depth_adjustment: float  # -2.0 to +2.0 meters
    viscosity_level: str  # 'very_low', 'low', 'baseline', 'high', 'very_high'
    scenario: Optional[str] = None  # 'drought', 'baseline', 'high_flow', 'dredged'

class TidalResults(BaseModel):
    """Output results from the model"""
    channels: Dict[str, List[Dict]]
    vertices: Dict[str, Dict]
    parameters_used: Dict[str, float]
    max_amplitude: float
    phase_lag: float

# Parameter mapping configurations
FRICTION_MAPPING = {
    'very_low': 0.002,   # Smooth bed, low vegetation
    'low': 0.003,        # Clean channel
    'baseline': 0.0047272727272727275,   # Current conditions (calibrated)
    'high': 0.007,       # Moderate vegetation
    'very_high': 0.010   # Dense vegetation/obstacles
}

VISCOSITY_MAPPING = {
    'very_low': 0.002,
    'low': 0.003,
    'baseline': 0.0059191919191919195,   # Current conditions (calibrated)
    'high': 0.007,
    'very_high': 0.010
}

SCENARIO_PRESETS = {
    'drought': {'Sf': 0.007, 'Av': 0.006, 'depth_adj': -1.5},
    'baseline': {'Sf': 0.0047272727272727275, 'Av': 0.0059191919191919195, 'depth_adj': 0.0},
    'high_flow': {'Sf': 0.004, 'Av': 0.004, 'depth_adj': +1.0},
    'dredged': {'Sf': 0.003, 'Av': 0.004, 'depth_adj': +2.0}
}

def complex_to_dict(z):
    """Convert complex number to JSON-serializable dict"""
    if isinstance(z, np.ndarray):
        return [{'real': float(x.real), 'imag': float(x.imag), 
                 'amplitude': float(abs(x)), 'phase': float(np.angle(x))} 
                for x in z]
    return {'real': float(z.real), 'imag': float(z.imag), 
            'amplitude': float(abs(z)), 'phase': float(np.angle(z))}

def complex_array_to_amplitude(arr):
    """Convert 3D complex array to 3D amplitude array (JSON-safe)"""
    if arr is None:
        return None
    # Convert complex to amplitude (magnitude)
    return np.abs(arr).tolist()

def process_channel_results(eta, u, x, branch_index=0, reverse_x=False, x_offset=0):
    """
    Extract results for a specific channel branch
    
    Parameters:
    - eta: water level amplitude array
    - u: velocity array
    - x: position array
    - branch_index: which branch to extract (for multi-branch arrays)
    - reverse_x: if True, reverse the x-axis (model uses ocean=high_x, reality is opposite)
    - x_offset: offset to add to x-coordinates for network alignment
    """
    results = []
    
    try:
        for i in range(len(eta)):
            # Handle different array dimensions
            if eta.ndim == 1:
                eta_val = eta[i]
                u_val = u[i] if u.ndim == 1 else u[i, 0]
                x_val = x[i] if x.ndim == 1 else x[i, 0]
            else:
                # Multi-branch case
                eta_val = eta[i, branch_index]
                u_val = u[i, branch_index] if u.ndim > 1 else u[i]
                x_val = x[i, branch_index] if x.ndim > 1 else x[i]
            
            # Apply offset for network alignment
            x_val_adjusted = x_val + x_offset
            
            results.append({
                'position': float(x_val_adjusted),
                'position_km': float(x_val_adjusted / 1000),  # Also provide in km
                'eta': complex_to_dict(eta_val),
                'velocity': complex_to_dict(u_val)
            })
        
        # Reverse the order if needed (to match real geography)
        if reverse_x:
            results = list(reversed(results))
            
    except Exception as e:
        print(f"Error processing branch {branch_index}: {e}")
        import traceback
        traceback.print_exc()
        # Return minimal valid data
        results = [{
            'position': 0.0,
            'position_km': 0.0,
            'eta': {'real': 0.0, 'imag': 0.0, 'amplitude': 0.0, 'phase': 0.0},
            'velocity': {'real': 0.0, 'imag': 0.0, 'amplitude': 0.0, 'phase': 0.0}
        }]
    
    return results

@app.get("/")
def read_root():
    return {
        "message": "Rhine-Meuse Tidal Model API",
        "version": "2.0 Enhanced",
        "new_features": [
            "M4 component breakdown (advection, no-stress, Stokes)",
            "Tidal asymmetry analysis",
            "Interactive hover tooltips support"
        ],
        "network": {
            "topology": {
                "v1": "Junction: Waal (WL) - Nieuwe Maas (NM) - Nieuwe Merwede (NE)",
                "v2": "Junction: Oude Maas (OM) - Nieuwe Merwede (NE) - Haringvliet (HV)",
                "v3": "Junction: Nieuwe Waterweg (NW) - Hartelkanaal (HK) - Nieuwe Maas (NM) - Oude Maas (OM)"
            },
            "ocean_channels": ["Nieuwe Waterweg (NW)", "Hartelkanaal (HK)"],
            "middle_channels": ["Nieuwe Maas (NM)", "Oude Maas (OM)", "Nieuwe Merwede (NE)"],
            "river_channels": ["Waal (WL)"],
            "closed_channels": ["Haringvliet (HV) - closed, tide dampening"],
            "note": "Model x-axis is reversed: ocean channels at x_max, river at x_min"
        },
        "endpoints": {
            "/run_model": "POST - Run tidal model with parameters",
            "/presets": "GET - Get available scenario presets",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "ready"}

@app.get("/presets")
def get_presets():
    """Return available scenario presets"""
    return {
        "friction_levels": list(FRICTION_MAPPING.keys()),
        "viscosity_levels": list(VISCOSITY_MAPPING.keys()),
        "scenarios": SCENARIO_PRESETS,
        "depth_range": {"min": -2.0, "max": 2.0, "unit": "meters"}
    }

@app.post("/run_model")
def run_tidal_model(params: ModelParameters):
    """
    Run the Rhine-Meuse tidal model with specified parameters
    Enhanced version with M4 breakdown and tidal asymmetry
    """
    try:
        # Apply scenario preset if provided
        if params.scenario and params.scenario in SCENARIO_PRESETS:
            preset = SCENARIO_PRESETS[params.scenario]
            Sf = preset['Sf']
            Av = preset['Av']
            depth_adj = preset['depth_adj']
        else:
            Sf = FRICTION_MAPPING.get(params.friction_level, 0.005)
            Av = VISCOSITY_MAPPING.get(params.viscosity_level, 0.005)
            depth_adj = params.depth_adjustment
        
        # Run M2 model
        print(f"Running model with Sf={Sf}, Av={Av}, depth_adj={depth_adj}")
        model = Network_model_RM(Av=Av, Sf=Sf)
        model.H_r = model.H_r + depth_adj
        model.H_m = model.H_m + depth_adj
        model.H_o = model.H_o + depth_adj
        model.M2()
        
        # Run M4 model
        print("Running M4 model...")
        model.M4_no_stress()
        
        # Process results for all channels with correct topology
        waal_results = process_channel_results(
            model.eta0_r, model.u0_r, model.x_r,
            branch_index=0, reverse_x=True, x_offset=0
        )
        
        haringvliet_results = process_channel_results(
            model.eta0_r, model.u0_r, model.x_r,
            branch_index=1, reverse_x=True, x_offset=0
        )
        
        nieuwe_maas_results = process_channel_results(
            model.eta0_m, model.u0_m, model.x_m,
            branch_index=0, reverse_x=True, x_offset=0
        )
        
        nieuwe_merwede_results = process_channel_results(
            model.eta0_m, model.u0_m, model.x_m,
            branch_index=1, reverse_x=True, x_offset=0
        )
        
        oude_maas_results = process_channel_results(
            model.eta0_m, model.u0_m, model.x_m,
            branch_index=2, reverse_x=True, x_offset=0
        )
        
        nieuwe_waterweg_results = process_channel_results(
            model.eta0_o, model.u0_o, model.x_o,
            branch_index=0, reverse_x=True, x_offset=0
        )
        
        hartelkanaal_results = process_channel_results(
            model.eta0_o, model.u0_o, model.x_o,
            branch_index=1, reverse_x=True, x_offset=0
        )
        
        # Map for easy lookup
        channel_results_map = {
            'waal': waal_results,
            'haringvliet': haringvliet_results,
            'nieuwe_maas': nieuwe_maas_results,
            'nieuwe_merwede': nieuwe_merwede_results,
            'oude_maas': oude_maas_results,
            'nieuwe_waterweg': nieuwe_waterweg_results,
            'hartelkanaal': hartelkanaal_results
        }
        
        channel_names = list(channel_results_map.keys())
        
        # Process M2 time series
        downsample = 10
        time_series = {}
        for ch_name in channel_names:
            channel_data = channel_results_map[ch_name]
            sampled_points = []
            for i in range(0, len(channel_data), downsample):
                pt = channel_data[i]
                sampled_points.append({
                    'position': pt['position'],
                    'position_km': pt['position_km'],
                    'amplitude': pt['eta']['amplitude'],
                    'phase': pt['eta']['phase'],
                    'index': i
                })
            
            time_series[ch_name] = {
                'points': sampled_points,
                'total_points': len(channel_data)
            }
        
        # Calculate global min/max for color scale
        all_amplitudes = []
        for ch_name in channel_names:
            for point in time_series[ch_name]['points']:
                all_amplitudes.append(point['amplitude'])
        
        global_min = -max(all_amplitudes)
        global_max = max(all_amplitudes)
        
        # Process M4 time series with component breakdown
        print("Processing M4 component breakdown...")
        time_series_m4 = {}
        m4_breakdown = {}
        
        for ch_name in channel_names:
            m2_channel_data = channel_results_map[ch_name]
            
            # Get M4 data from the model
            if ch_name in ['nieuwe_waterweg', 'hartelkanaal']:
                eta_m4_full = model.eta4_o
                section = 'ocean'
                branch_map = {'nieuwe_waterweg': 0, 'hartelkanaal': 1}
                branch_idx = branch_map[ch_name]
            elif ch_name in ['nieuwe_maas', 'nieuwe_merwede', 'oude_maas']:
                eta_m4_full = model.eta4_m
                section = 'middle'
                branch_map = {'nieuwe_maas': 0, 'nieuwe_merwede': 1, 'oude_maas': 2}
                branch_idx = branch_map[ch_name]
            else:  # river
                eta_m4_full = model.eta4_r
                section = 'river'
                branch_map = {'waal': 0, 'haringvliet': 1}
                branch_idx = branch_map[ch_name]
            
            sampled_points_m4 = []
            breakdown_points = []
            
            for idx, i in enumerate(range(0, len(m2_channel_data), downsample)):
                if i >= len(eta_m4_full):
                    break
                
                m2_pt = m2_channel_data[i]
                
                # Extract M4 value
                if eta_m4_full.ndim == 1:
                    eta_val_m4 = eta_m4_full[i]
                else:
                    eta_val_m4 = eta_m4_full[i, branch_idx]
                
                # Get M4 component breakdown
                breakdown_data = {
                    'position': m2_pt['position'],
                    'position_km': m2_pt['position_km']
                }
                
                # Try to get breakdown components if available
                try:
                    if section == 'ocean' and hasattr(model, 'eta_A_o'):
                        eta_A = model.eta_A_o[i, branch_idx] if model.eta_A_o.ndim > 1 else model.eta_A_o[i]
                        eta_N = model.eta_N_o[i, branch_idx] if model.eta_N_o.ndim > 1 else model.eta_N_o[i]
                        eta_S = model.eta_S_o[i, branch_idx] if model.eta_S_o.ndim > 1 else model.eta_S_o[i]
                        breakdown_data.update({
                            'advection': float(np.abs(eta_A)),
                            'no_stress': float(np.abs(eta_N)),
                            'stokes': float(np.abs(eta_S))
                        })
                    elif section == 'middle' and hasattr(model, 'eta_A_m'):
                        eta_A = model.eta_A_m[i, branch_idx] if model.eta_A_m.ndim > 1 else model.eta_A_m[i]
                        eta_N = model.eta_N_m[i, branch_idx] if model.eta_N_m.ndim > 1 else model.eta_N_m[i]
                        eta_S = model.eta_S_m[i, branch_idx] if model.eta_S_m.ndim > 1 else model.eta_S_m[i]
                        breakdown_data.update({
                            'advection': float(np.abs(eta_A)),
                            'no_stress': float(np.abs(eta_N)),
                            'stokes': float(np.abs(eta_S))
                        })
                    elif section == 'river' and hasattr(model, 'eta_A_r'):
                        eta_A = model.eta_A_r[i, branch_idx] if model.eta_A_r.ndim > 1 else model.eta_A_r[i]
                        eta_N = model.eta_N_r[i, branch_idx] if model.eta_N_r.ndim > 1 else model.eta_N_r[i]
                        eta_S = model.eta_S_r[i, branch_idx] if model.eta_S_r.ndim > 1 else model.eta_S_r[i]
                        breakdown_data.update({
                            'advection': float(np.abs(eta_A)),
                            'no_stress': float(np.abs(eta_N)),
                            'stokes': float(np.abs(eta_S))
                        })
                except Exception as e:
                    print(f"Warning: Could not extract M4 breakdown for {ch_name}: {e}")
                
                breakdown_points.append(breakdown_data)
                
                sampled_points_m4.append({
                    'position': m2_pt['position'],
                    'position_km': m2_pt['position_km'],
                    'amplitude': float(np.abs(eta_val_m4)),
                    'phase': float(np.angle(eta_val_m4)),
                    'index': i
                })
            
            time_series_m4[ch_name] = {
                'points': sampled_points_m4,
                'total_points': len(m2_channel_data)
            }
            
            m4_breakdown[ch_name] = {
                'points': breakdown_points,
                'has_data': len(breakdown_points) > 0 and 'advection' in breakdown_points[0]
            }
        
        # Calculate M4/M2 ratio and tidal asymmetry
        print("Calculating tidal asymmetry...")
        m4_m2_ratio = {}
        tidal_asymmetry = {}
        
        for ch_name in channel_names:
            m2_full_data = channel_results_map[ch_name]
            m4_points = time_series_m4[ch_name]['points']
            
            ratio_points = []
            asymmetry_points = []
            
            for idx, i in enumerate(range(0, len(m2_full_data), downsample)):
                if idx >= len(m4_points):
                    break
                    
                m2_pt = m2_full_data[i]
                m4_pt = m4_points[idx]
                
                m2_amp = m2_pt['eta']['amplitude']
                m4_amp = m4_pt['amplitude']
                m2_phase = m2_pt['eta']['phase']
                m4_phase = m4_pt['phase']
                
                if m2_amp > 0.001:
                    ratio = m4_amp / m2_amp
                else:
                    ratio = 0.0
                
                # Tidal asymmetry metrics
                phase_diff = m4_phase - 2 * m2_phase
                while phase_diff > np.pi:
                    phase_diff -= 2 * np.pi
                while phase_diff < -np.pi:
                    phase_diff += 2 * np.pi
                
                skewness = ratio * np.cos(phase_diff)
                asymmetry_strength = 2 * ratio / (1 + ratio**2) if ratio > 0 else 0
                
                ratio_points.append({
                    'position': m2_pt['position'],
                    'position_km': m2_pt['position_km'],
                    'ratio': float(ratio),
                    'index': i
                })
                
                asymmetry_points.append({
                    'position': m2_pt['position'],
                    'position_km': m2_pt['position_km'],
                    'skewness': float(skewness),
                    'phase_difference': float(phase_diff * 180 / np.pi),
                    'asymmetry_strength': float(asymmetry_strength),
                    'flood_dominant': bool(skewness > 0),
                    'index': i
                })
            
            m4_m2_ratio[ch_name] = {
                'points': ratio_points,
                'total_points': len(m2_full_data)
            }
            
            tidal_asymmetry[ch_name] = {
                'points': asymmetry_points,
                'total_points': len(m2_full_data)
            }
        
        # Calculate global min/max
        all_amplitudes_m4 = [pt['amplitude'] for ch in channel_names for pt in time_series_m4[ch]['points']]
        all_ratios = [pt['ratio'] for ch in channel_names for pt in m4_m2_ratio[ch]['points']]
        all_skewness = [pt['skewness'] for ch in channel_names for pt in tidal_asymmetry[ch]['points']]
        
        global_min_m4 = -max(all_amplitudes_m4)
        global_max_m4 = max(all_amplitudes_m4)
        global_min_ratio = min(all_ratios) if all_ratios else 0
        global_max_ratio = max(all_ratios) if all_ratios else 1
        global_min_skewness = min(all_skewness) if all_skewness else -1
        global_max_skewness = max(all_skewness) if all_skewness else 1
        
        # Compile results
        results = {
            "channels": channel_results_map,
            "vertices": {
                "v1": complex_to_dict(model.eta_vertex_1),
                "v2": complex_to_dict(model.eta_vertex_2),
                "v3": complex_to_dict(model.eta_vertex_3)
            },
            "parameters_used": {
                "Sf": float(Sf),
                "Av": float(Av),
                "depth_adjustment": float(depth_adj)
            },
            "max_amplitude": float(np.max(np.abs(model.eta0_r))),
            "phase_lag": float(np.angle(model.eta0_r[-1, 0] if model.eta0_r.ndim > 1 else model.eta0_r[-1])),
            "time_series": {
                "data": time_series,
                "num_frames": 60,
                "period_hours": 12.42,
                "format": "spatial_points",
                "global_min": float(global_min),
                "global_max": float(global_max),
                "description": "M2 tidal component"
            },
            "time_series_m4": {
                "data": time_series_m4,
                "num_frames": 60,
                "period_hours": 6.21,
                "format": "spatial_points",
                "global_min": float(global_min_m4),
                "global_max": float(global_max_m4),
                "description": "M4 tidal component"
            },
            "m4_breakdown": {
                "data": m4_breakdown,
                "description": "M4 component breakdown: advection, no-stress, and Stokes"
            },
            "m4_m2_ratio": {
                "data": m4_m2_ratio,
                "global_min": float(global_min_ratio),
                "global_max": float(global_max_ratio),
                "description": "M4/M2 amplitude ratio"
            },
            "tidal_asymmetry": {
                "data": tidal_asymmetry,
                "global_min_skewness": float(global_min_skewness),
                "global_max_skewness": float(global_max_skewness),
                "description": "Tidal asymmetry metrics (skewness, phase difference, flood/ebb dominance)"
            },
            "velocity_structure": {
                "ocean": {
                    "u0": complex_array_to_amplitude(model.u0_o[::2, ::2, :]) if hasattr(model, 'u0_o') else None,
                    "x": (model.x_o[::2, 0].tolist() if model.x_o.ndim > 1 else model.x_o[::2].tolist()) if hasattr(model, 'x_o') else None,
                    "z": (model.z_o[::2, 0].tolist() if model.z_o.ndim > 1 else model.z_o[::2].tolist()) if hasattr(model, 'z_o') else None,
                    "channels": ["nieuwe_waterweg", "hartelkanaal"]
                },
                "middle": {
                    "u0": complex_array_to_amplitude(model.u0_m[::2, ::2, :]) if hasattr(model, 'u0_m') else None,
                    "x": (model.x_m[::2, 0].tolist() if model.x_m.ndim > 1 else model.x_m[::2].tolist()) if hasattr(model, 'x_m') else None,
                    "z": (model.z_m[::2, 0].tolist() if model.z_m.ndim > 1 else model.z_m[::2].tolist()) if hasattr(model, 'z_m') else None,
                    "channels": ["nieuwe_maas", "nieuwe_merwede", "oude_maas"]
                },
                "river": {
                    "u0": complex_array_to_amplitude(model.u0_r[::2, ::2, :]) if hasattr(model, 'u0_r') else None,
                    "x": (model.x_r[::2, 0].tolist() if model.x_r.ndim > 1 else model.x_r[::2].tolist()) if hasattr(model, 'x_r') else None,
                    "z": (model.z_r[::2, 0].tolist() if model.z_r.ndim > 1 else model.z_r[::2].tolist()) if hasattr(model, 'z_r') else None,
                    "channels": ["waal", "haringvliet"]
                },
                "description": "3D velocity amplitude for M2 tide"
            }
        }
        
        print("Results processed successfully with M4 breakdown and tidal asymmetry")
        return results
        
    except Exception as e:
        print(f"ERROR in run_model: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model execution failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)