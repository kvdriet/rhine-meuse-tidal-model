from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import json

# Import your model class
from rhine_meuse_test import Network_model_RM

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
        
        # Adjust depths for all channel types
        model.H_r = model.H_r + depth_adj  # River channels (includes Waal + Haringvliet)
        model.H_m = model.H_m + depth_adj  # Middle channels (OM, NM, NE)
        model.H_o = model.H_o + depth_adj  # Ocean channels (NW, HK)
        
        # Run M2 tide calculation
        model.M2()
        
        print("Model run complete, processing results...")
        
        # Calculate positions for natural geography (ocean LEFT, river RIGHT)
        # We'll invert the coordinate system: multiply all positions by -1
        # This makes ocean channels appear at negative x (left side)
        # and river channels at positive x (right side)
        
        C = float(np.max(model.L_r)) / 1000  # km, reference point
        
        # River positions
        x_river_begin = (np.max(model.L_r) - model.L_r) / 1000  # km
        test = np.array([0, (model.L_m[1] + model.L_r[1]) / 1000])  # km
        
        # For natural geography: negate all positions after applying offsets
        # This inverts left-right so ocean is on the left
        
        # Ocean channels (will appear on the LEFT, negative x)
        nieuwe_waterweg_results = process_channel_results(model.eta0_o, model.u0_mean_o, model.x_o, 
                                                          branch_index=0, reverse_x=True,
                                                          x_offset=-C * 1000)
        hartelkanaal_results = process_channel_results(model.eta0_o, model.u0_mean_o, model.x_o, 
                                                       branch_index=1, reverse_x=True,
                                                       x_offset=-C * 1000)
        
        # Negate x-coordinates for natural geography
        for result in nieuwe_waterweg_results:
            result['position'] = -result['position']
            result['position_km'] = -result['position_km']
        for result in hartelkanaal_results:
            result['position'] = -result['position']
            result['position_km'] = -result['position_km']
        
        # Middle channels
        # Oude Maas should align with ocean endpoint, others offset normally
        nieuwe_maas_results = process_channel_results(model.eta0_m, model.u0_mean_m, model.x_m, 
                                                      branch_index=0, reverse_x=True,
                                                      x_offset=-C * 1000)
        nieuwe_merwede_results = process_channel_results(model.eta0_m, model.u0_mean_m, model.x_m, 
                                                         branch_index=1, reverse_x=True,
                                                         x_offset=-C * 1000)
        # Oude Maas: special handling to align with ocean
        oude_maas_results = process_channel_results(model.eta0_m, model.u0_mean_m, model.x_m, 
                                                    branch_index=2, reverse_x=True,
                                                    x_offset=-C * 1000)
        
        # Calculate ocean endpoint position
        if nieuwe_waterweg_results:
            ocean_end_pos = max([r['position'] for r in nieuwe_waterweg_results + hartelkanaal_results])
            # Shift Oude Maas to start at ocean endpoint
            if oude_maas_results:
                oude_maas_start = oude_maas_results[0]['position']
                shift = ocean_end_pos - oude_maas_start
                for result in oude_maas_results:
                    result['position'] += shift
                    result['position_km'] += shift / 1000
        
        # Negate for natural geography
        for result in nieuwe_maas_results:
            result['position'] = -result['position']
            result['position_km'] = -result['position_km']
        for result in nieuwe_merwede_results:
            result['position'] = -result['position']
            result['position_km'] = -result['position_km']
        for result in oude_maas_results:
            result['position'] = -result['position']
            result['position_km'] = -result['position_km']
        
        # River channels (will appear on the RIGHT, positive x)
        waal_offset = (x_river_begin[0] + test[0] - C) * 1000
        waal_results = process_channel_results(model.eta0_r, model.u0_mean_r, model.x_r, 
                                              branch_index=0, reverse_x=True, 
                                              x_offset=waal_offset)
        
        # Haringvliet - REVERSE the amplitude array since flow is opposite
        haringvliet_offset = (x_river_begin[1] + test[1] - C) * 1000
        haringvliet_results = process_channel_results(model.eta0_r, model.u0_mean_r, model.x_r, 
                                                     branch_index=1, reverse_x=False,  # Don't reverse
                                                     x_offset=haringvliet_offset)
        
        # Negate for natural geography
        for result in waal_results:
            result['position'] = -result['position']
            result['position_km'] = -result['position_km']
        for result in haringvliet_results:
            result['position'] = -result['position']
            result['position_km'] = -result['position_km']
        
        # Reverse haringvliet results order since flow is opposite
        haringvliet_results = list(reversed(haringvliet_results))
        
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
            "phase_lag": float(np.angle(model.eta0_r[-1, 0] if model.eta0_r.ndim > 1 else model.eta0_r[-1]))
        }
        
        print("Results processed successfully for all 7 channels")
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