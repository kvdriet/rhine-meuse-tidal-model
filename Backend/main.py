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
        "version": "1.0",
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
    
    Network topology:
    - River: Waal (WL, branch 0 of eta0_r)
    - River-like (closed): Haringvliet (HV, branch 1 of eta0_r - branches from v2)
    - Middle: 
      * Nieuwe Maas (NM, branch 0 of eta0_m) - connects v2-v3 (ocean)
      * Nieuwe Merwede (NE, branch 1 of eta0_m) - connects v1-v2
      * Oude Maas (OM, branch 2 of eta0_m) - connects v2-v3 (ocean)
    - Ocean: 
      * Nieuwe Waterweg (NW, branch 0 of eta0_o)
      * Hartelkanaal (HK, branch 1 of eta0_o)
    
    Vertices:
    - v1: Waal (WL) - Nieuwe Merwede (NE) junction
    - v2: Nieuwe Maas (NM) - Nieuwe Merwede (NE) - Oude Maas (OM) - Haringvliet (HV) junction
    - v3: Nieuwe Waterweg (NW) - Hartelkanaal (HK) - Nieuwe Maas (NM) - Oude Maas (OM) junction
    
    Note: Model x-axis is reversed (ocean at high x, river at low x)
    """
    try:
        # Apply scenario preset if specified
        if params.scenario and params.scenario in SCENARIO_PRESETS:
            preset = SCENARIO_PRESETS[params.scenario]
            Sf = preset['Sf']
            Av = preset['Av']
            depth_adj = preset['depth_adj']
        else:
            # Use individual parameters
            Sf = FRICTION_MAPPING.get(params.friction_level, 0.005)
            Av = VISCOSITY_MAPPING.get(params.viscosity_level, 0.005)
            depth_adj = params.depth_adjustment
        
        print(f"Running model with Av={Av}, Sf={Sf}, depth_adj={depth_adj}")
        
        # Initialize model with parameters
        model = Network_model_RM(Av=Av, Sf=Sf)
        
        # OPTIMIZATION: Reduce spatial resolution to save memory
        # This allows more animation frames for smoother propagation visualization
        model.N = 100  # Reduced from 500 to 100 (5x memory reduction)
        
        # Recreate spatial grids with new N
        model.x = np.linspace(0, model.L, model.N)
        model.x_r = np.linspace(0, model.L_r, model.N)
        model.x_m1 = np.linspace(model.L_r[0], model.L_r[0] + model.L_m[:2], model.N)
        model.x_m2 = np.linspace(model.L_r[0] + model.L_m[1], model.L_r[0] + model.L_m[1] + model.L_m[2], model.N)
        model.x_m = np.column_stack((model.x_m1, model.x_m2))
        model.x_o = np.linspace(model.L_r[0] + model.L_m[0], model.L, model.N)
        model.z_r = np.linspace(-model.H_r, 0, model.N)
        model.z_o = np.linspace(-model.H_o, 0, model.N)
        model.z_m = np.linspace(-model.H_m, 0, model.N)
        model.B_rx = model.B_river * np.exp((model.x_r - model.x_r[0]) / model.Lb_r)
        model.B_mx = model.B_middle * np.exp((model.x_m - model.x_m[0]) / model.Lb_m)
        model.B_ox = model.B_ocean * np.exp((model.x_o - model.x_o[0]) / model.Lb_o)
        
        # Adjust depths for all channel types
        model.H_r = model.H_r + depth_adj  # River channels (includes Waal + Haringvliet)
        model.H_m = model.H_m + depth_adj  # Middle channels (OM, NM, NE)
        model.H_o = model.H_o + depth_adj  # Ocean channels (NW, HK)
        
        # Run M2 tide calculation
        model.M2()
        
        # Store M2 results
        eta0_m2_river = model.eta0_r.copy()
        eta0_m2_ocean = model.eta0_o.copy()
        eta0_m2_middle = model.eta0_m.copy()
        
        # Run M4 tide calculation
        model.M2(M4="yes")
        
        # Store M4 results
        eta0_m4_river = model.eta0_r.copy()
        eta0_m4_ocean = model.eta0_o.copy()
        eta0_m4_middle = model.eta0_m.copy()
        
        # Calculate M4/M2 ratio
        eta0_ratio_river = np.abs(eta0_m4_river) / (np.abs(eta0_m2_river) + 1e-10)  # Add small value to avoid division by zero
        eta0_ratio_ocean = np.abs(eta0_m4_ocean) / (np.abs(eta0_m2_ocean) + 1e-10)
        eta0_ratio_middle = np.abs(eta0_m4_middle) / (np.abs(eta0_m2_middle) + 1e-10)
        
        # Restore M2 results to model for main visualization
        model.eta0_r = eta0_m2_river
        model.eta0_o = eta0_m2_ocean
        model.eta0_m = eta0_m2_middle

        eta0_m2_ocean = model.eta0_o.copy()
        eta0_m2_middle = model.eta0_m.copy()
        
        # Run M4 tide calculation
        model.M2(M4="yes")
        
        # Store M4 results
        eta0_m4_river = model.eta0_r.copy()
        eta0_m4_ocean = model.eta0_o.copy()
        eta0_m4_middle = model.eta0_m.copy()
        
        # Restore M2 for main results (backwards compatibility)
        model.eta0_r = eta0_m2_river
        model.eta0_o = eta0_m2_ocean
        model.eta0_m = eta0_m2_middle
        
        print("Model run complete (M2 and M4), processing results...")
        
        # Using the working Python plotting logic (ocean right, river left), then flipping
        # This matches your original matplotlib code that works correctly
        
        C = float(np.max(model.L_r)) / 1000  # km, reference point (95 km)
        
        # Calculate river x positions
        x_river_begin = (np.max(model.L_r) - model.L_r) / 1000  # km
        test = np.array([0, (model.L_m[1] + model.L_r[1]) / 1000])  # km
        
        # OCEAN CHANNELS - process with offset -C (will be on right before flip)
        nieuwe_waterweg_results = process_channel_results(model.eta0_o, model.u0_mean_o, model.x_o, 
                                                          branch_index=0, reverse_x=False,
                                                          x_offset=-C * 1000)
        hartelkanaal_results = process_channel_results(model.eta0_o, model.u0_mean_o, model.x_o, 
                                                       branch_index=1, reverse_x=False,
                                                       x_offset=-C * 1000)
        
        # MIDDLE CHANNELS - NM and NE standard, OM needs special x_middle3 treatment
        nieuwe_maas_results = process_channel_results(model.eta0_m, model.u0_mean_m, model.x_m, 
                                                      branch_index=0, reverse_x=False,
                                                      x_offset=-C * 1000)
        nieuwe_merwede_results = process_channel_results(model.eta0_m, model.u0_mean_m, model.x_m, 
                                                         branch_index=1, reverse_x=False,
                                                         x_offset=-C * 1000)
        
        # Oude Maas (OM) - create x_middle3 positions (from x_middle[-1,1] to x_ocean[0,0])
        # This connects OM properly between middle and ocean
        oude_maas_results = []
        N = len(model.eta0_m)
        x_middle3_start = model.x_m[-1, 1]  # End of middle branch 1 (NE)
        x_middle3_end = model.x_o[0, 0]  # Start of ocean branch 0
        x_middle3 = np.linspace(x_middle3_start, x_middle3_end, N)
        
        for i in range(N):
            oude_maas_results.append({
                'position': float(x_middle3[i] - C * 1000),
                'position_km': float(x_middle3[i] / 1000 - C),
                'eta': complex_to_dict(model.eta0_m[i, 2]),
                'velocity': complex_to_dict(model.u0_mean_m[i, 2] if model.u0_mean_m.ndim > 1 else model.u0_mean_m[i])
            })
        
        # RIVER CHANNELS
        # Waal
        waal_results = []
        for i in range(len(model.x_r)):
            x_val = (x_river_begin[0] + test[0]) + model.x_r[i, 0] / 1000
            waal_results.append({
                'position': float((x_val - C) * 1000),
                'position_km': float(x_val - C),
                'eta': complex_to_dict(model.eta0_r[i, 0]),
                'velocity': complex_to_dict(model.u0_mean_r[i, 0])
            })
        
        # Haringvliet - with reversed amplitudes ([::-1])
        haringvliet_results = []
        for i in range(len(model.x_r)):
            x_val = (x_river_begin[1] + test[1]) + model.x_r[i, 1] / 1000
            # Reverse the amplitude index
            rev_idx = len(model.x_r) - 1 - i
            haringvliet_results.append({
                'position': float((x_val - C) * 1000),
                'position_km': float(x_val - C),
                'eta': complex_to_dict(model.eta0_r[rev_idx, 1]),  # Reversed!
                'velocity': complex_to_dict(model.u0_mean_r[rev_idx, 1])  # Reversed!
            })
        
        # NOW FLIP EVERYTHING: Negate all x-coordinates to put ocean on LEFT, river on RIGHT
        all_results = [
            nieuwe_waterweg_results, hartelkanaal_results,
            nieuwe_maas_results, nieuwe_merwede_results, oude_maas_results,
            waal_results, haringvliet_results
        ]
        
        for results_list in all_results:
            for result in results_list:
                result['position'] = -result['position']
                result['position_km'] = -result['position_km']
        
        # SPATIAL ANIMATION: Send points along each channel with coordinates
        # This matches your Python scatter plot approach
        time_series = {}
        channel_names = ['nieuwe_waterweg', 'hartelkanaal', 'haringvliet', 
                        'nieuwe_maas', 'nieuwe_merwede', 'oude_maas', 'waal']
        
        # Map channels to their result arrays (which contain coordinates)
        channel_results_map = {
            'nieuwe_waterweg': nieuwe_waterweg_results,
            'hartelkanaal': hartelkanaal_results,
            'nieuwe_maas': nieuwe_maas_results,
            'nieuwe_merwede': nieuwe_merwede_results,
            'oude_maas': oude_maas_results,
            'haringvliet': haringvliet_results,
            'waal': waal_results
        }
        
        # For each channel, extract spatial points (sample every 2nd point for smooth line effect)
        downsample = 2  # Was 5, now 2 for smoother gradient line
        for ch_name in channel_names:
            results = channel_results_map[ch_name]
            
            # Get coordinates from KML data (we need to match result indices to coordinates)
            # Sample points for animation
            sampled_points = []
            for i in range(0, len(results), downsample):
                eta_complex = results[i]['eta']
                eta_val = complex(eta_complex['real'], eta_complex['imag'])
                
                sampled_points.append({
                    'amplitude': float(np.abs(eta_val)),
                    'phase': float(np.angle(eta_val)),
                    'index': i  # Index to match with coordinates in frontend
                })
            
            time_series[ch_name] = {
                'points': sampled_points,
                'total_points': len(results)
            }
        
        # Calculate global min/max for consistent color scaling (like your vmin/vmax)
        all_amplitudes = []
        for ch_name in channel_names:
            for point in time_series[ch_name]['points']:
                all_amplitudes.append(point['amplitude'])
        
        global_min = -max(all_amplitudes)  # For vmin (negative of max amplitude)
        global_max = max(all_amplitudes)   # For vmax
        
        # Create M4 data (using stored M4 results)
        time_series_m4 = {}
        m4_channel_data_map = {
            'nieuwe_waterweg': (eta0_m4_ocean, 0),
            'hartelkanaal': (eta0_m4_ocean, 1),
            'nieuwe_maas': (eta0_m4_middle, 0),
            'nieuwe_merwede': (eta0_m4_middle, 1),
            'oude_maas': (eta0_m4_middle, 2),
            'haringvliet': (eta0_m4_river, 1),
            'waal': (eta0_m4_river, 0)
        }
        
        for ch_name in channel_names:
            eta_array_m4, branch_idx = m4_channel_data_map[ch_name]
            if eta_array_m4.ndim > 1:
                eta_channel_m4 = eta_array_m4[:, branch_idx]
            else:
                eta_channel_m4 = eta_array_m4
            
            # Sample every downsample points
            sampled_points_m4 = []
            for i in range(0, len(eta_channel_m4), downsample):
                eta_val_m4 = eta_channel_m4[i]
                sampled_points_m4.append({
                    'amplitude': float(np.abs(eta_val_m4)),
                    'phase': float(np.angle(eta_val_m4)),
                    'index': i
                })
            
            time_series_m4[ch_name] = {
                'points': sampled_points_m4,
                'total_points': len(eta_channel_m4)
            }
        
        # Calculate M4/M2 ratio for each channel
        m4_m2_ratio = {}
        for ch_name in channel_names:
            # Get M2 and M4 amplitudes along channel
            m2_amps = [p['amplitude'] for p in time_series[ch_name]['points']]
            m4_amps = [p['amplitude'] for p in time_series_m4[ch_name]['points']]
            
            # Calculate ratio at each point
            ratio_points = []
            for i in range(len(m2_amps)):
                if m2_amps[i] > 0.001:  # Avoid division by zero
                    ratio = m4_amps[i] / m2_amps[i]
                else:
                    ratio = 0.0
                ratio_points.append({
                    'ratio': float(ratio),
                    'index': time_series[ch_name]['points'][i]['index']
                })
            
            m4_m2_ratio[ch_name] = {
                'points': ratio_points,
                'total_points': time_series[ch_name]['total_points']
            }
        
        # Calculate global min/max for M4
        all_amplitudes_m4 = []
        for ch_name in channel_names:
            for point in time_series_m4[ch_name]['points']:
                all_amplitudes_m4.append(point['amplitude'])
        
        global_min_m4 = -max(all_amplitudes_m4)
        global_max_m4 = max(all_amplitudes_m4)
        
        # Calculate global min/max for M4/M2 ratio
        all_ratios = []
        for ch_name in channel_names:
            for point in m4_m2_ratio[ch_name]['points']:
                all_ratios.append(point['ratio'])
        
        global_min_ratio = min(all_ratios) if all_ratios else 0
        global_max_ratio = max(all_ratios) if all_ratios else 1
        
        # Process results
        results = {
            "channels": {
                # River channels
                "waal": waal_results,
                "haringvliet": haringvliet_results,
                
                # Middle channels (correct order: NM, NE, OM)
                "nieuwe_maas": nieuwe_maas_results,
                "nieuwe_merwede": nieuwe_merwede_results,
                "oude_maas": oude_maas_results,
                
                # Ocean channels
                "nieuwe_waterweg": nieuwe_waterweg_results,
                "hartelkanaal": hartelkanaal_results
            },
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
                "num_frames": 60,  # Increased from 24 to 60 for smoother propagation
                "period_hours": 12.42,  # M2 tide period
                "format": "spatial_points",  # Spatial points with amplitude & phase
                "global_min": float(global_min),  # For color scale (vmin)
                "global_max": float(global_max),  # For color scale (vmax)
                "description": "Spatial points along each channel with amplitude/phase (like Python scatter plot)"
            },
            "time_series_m4": {
                "data": time_series_m4,
                "num_frames": 60,
                "period_hours": 6.21,  # M4 tide period (half of M2)
                "format": "spatial_points",
                "global_min": float(global_min_m4),
                "global_max": float(global_max_m4),
                "description": "M4 tidal component"
            },
            "m4_m2_ratio": {
                "data": m4_m2_ratio,
                "global_min": float(global_min_ratio),
                "global_max": float(global_max_ratio),
                "description": "M4/M2 amplitude ratio showing non-linear effects"
            }
        }
        
        print("Results processed successfully for all 7 channels with time series")
        return results
        
    except Exception as e:
        print(f"Error in model execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Model execution failed: {str(e)}")

@app.post("/run_model_quick")
def run_model_quick(params: ModelParameters):
    """
    Lightweight endpoint returning only key metrics for fast updates
    """
    try:
        Sf = FRICTION_MAPPING.get(params.friction_level, 0.005)
        Av = VISCOSITY_MAPPING.get(params.viscosity_level, 0.005)
        depth_adj = params.depth_adjustment
        
        # Run quick model
        model = Network_model_RM(Av=Av, Sf=Sf)
        model.H_r = model.H_r + depth_adj
        model.H_m = model.H_m + depth_adj
        model.H_o = model.H_o + depth_adj
        model.M2()
        
        # Return simplified results for real-time slider updates
        return {
            "max_amplitude": float(np.max(np.abs(model.eta0_r))),
            "phase_lag": float(np.angle(model.eta0_r[-1, 0] if model.eta0_r.ndim > 1 else model.eta0_r[-1])),
            "friction": float(Sf),
            "viscosity": float(Av),
            "success": True
        }
    except Exception as e:
        print(f"Error in quick model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)