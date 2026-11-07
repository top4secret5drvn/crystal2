import argparse
from pathlib import Path
from . import create_app
from .lease import WorkspaceLease


def main():
    parser=argparse.ArgumentParser(description='Crystal Lab — loopback research server')
    parser.add_argument('--workspace',type=Path,default=Path('lab-data'))
    parser.add_argument('--port',type=int,default=5050)
    args=parser.parse_args()
    with WorkspaceLease(args.workspace):
        app=create_app(args.workspace)
        print(f'Crystal Lab: http://127.0.0.1:{args.port} | workspace: {args.workspace.resolve()}')
        app.run(host='127.0.0.1',port=args.port,debug=False,use_reloader=False,threaded=True)

if __name__=='__main__':main()
