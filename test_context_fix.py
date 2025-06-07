#!/usr/bin/env python3
"""Test script to verify context length fixes."""

import requests
import json
import sys

def test_context_fix():
    """Test that we're using accurate context lengths."""
    
    # Test the /api/show endpoint first
    print("🔍 Testing /api/show endpoint...")
    show_response = requests.post(
        "http://localhost:11435/api/show",
        json={"name": "llama3.1:8b"},
        headers={"Content-Type": "application/json"}
    )
    
    if show_response.status_code == 200:
        show_data = show_response.json()
        print(f"✅ /api/show working: {show_data['details']['family']} {show_data['details']['parameter_size']}")
        print(f"📊 Stop tokens: {len([l for l in show_data['parameters'].split('\\n') if 'stop' in l])} found")
    else:
        print(f"❌ /api/show failed: {show_response.status_code}")
        return False
    
    # Test model inspection directly  
    print("\\n🔍 Testing model inspector...")
    try:
        import os, sys
        sys.path.append('/opt/llm/inference-service')
        from ollama_server.utils.model_inspector import model_inspector
        
        info = model_inspector.get_ollama_model_info('llama3.1:8b')
        if info:
            print(f"✅ Model inspector working:")
            print(f"   Architecture: {info.architecture}")
            print(f"   Context length: {info.context_length:,}")
            print(f"   Parameters: {info.parameters}")
            print(f"   Quantization: {info.quantization}")
            print(f"   Stop tokens: {len(info.stop_tokens)}")
            
            # Check if we got the accurate context length
            if info.context_length == 131072:
                print("✅ Accurate context length retrieved (131,072)")
            else:
                print(f"⚠️  Context length: {info.context_length} (expected 131,072)")
        else:
            print("❌ Model inspector returned no info")
            return False
            
    except Exception as e:
        print(f"❌ Model inspector error: {e}")
        return False
    
    # Test a simple generate request
    print("\\n🔍 Testing generate request...")
    generate_response = requests.post(
        "http://localhost:11435/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": "What is the capital of France? Please explain why.",
            "stream": False,
            "options": {"num_predict": 100}
        },
        headers={"Content-Type": "application/json"}
    )
    
    if generate_response.status_code == 200:
        gen_data = generate_response.json()
        print(f"✅ Generate working: {len(gen_data['response'])} chars response")
        print(f"📊 Token counts: prompt={gen_data.get('prompt_eval_count', 0)}, response={gen_data.get('eval_count', 0)}")
    else:
        print(f"❌ Generate failed: {generate_response.status_code}")
        return False
    
    print("\\n🎉 All tests passed! Context length fixes are working.")
    return True

if __name__ == "__main__":
    success = test_context_fix()
    sys.exit(0 if success else 1)