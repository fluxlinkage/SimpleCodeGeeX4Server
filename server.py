import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        self.send_response(405)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write("Method Not Allowed".encode('utf-8'))
    def do_POST(self):
        path = str(self.path)
        if path == '/v1/completions':
            global device;
            global tokenizer;
            global model;
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data)
            max_tokens=int(data['max_tokens'])
            prompt=data['prompt']
            inputs = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=False, tokenize=True, return_tensors="pt", return_dict=True).to(device)
            gen_kwargs = {'max_new_tokens': max_tokens}
            outputs = model.generate(**inputs,**gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            data = {
                'choices':[{'text':text}]
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write("Not Found".encode('utf-8'))

def add_code_generation_args(parser):
    group = parser.add_argument_group(title="CodeGeeX4 DEMO")
    group.add_argument(
        "--model-path",
        type=str,
        default="THUDM/codegeex4-all-9b",
    )
    # group.add_argument(
        # "--quantize",
        # type=int,
        # default=None,
    # )
    group.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    group.add_argument(
        "--cpu",
        action="store_true",
    )
    group.add_argument(
        "--listen",
        type=str,
        default="127.0.0.1",
    )
    group.add_argument(
        "--port",
        type=int,
        default=1911,
    )
    return parser
    
def get_model(args):
    global device;
    global tokenizer;
    global model;
    
    if not args.cpu:
        if torch.cuda.is_available():
            device = f"cuda:{args.gpu}"
        elif torch.backends.mps.is_built():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    #model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
    #return tokenizer, model

def main():
    
    parser = argparse.ArgumentParser()
    parser = add_code_generation_args(parser)
    args, _ = parser.parse_known_args()

    get_model(args)
    
    server_address = (args.listen, args.port)
    httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
    print(f'Starting httpd server on port {server_address[1]}')
    httpd.serve_forever()

if __name__ == '__main__':
    with torch.no_grad():
        main()