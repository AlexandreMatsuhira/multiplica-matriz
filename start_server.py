#!/usr/bin/env python3
"""
Script para iniciar servidor de computação Pyro
"""
import argparse
import Pyro5.api
from servers.computation_server import ComputationServer


def main():
    parser = argparse.ArgumentParser(
        description='Iniciar servidor de computação distribuída'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='server',
        help='Nome do servidor'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host do servidor (padrão: 0.0.0.0)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=9090,
        help='Porta do servidor (padrão: 9090)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"SERVIDOR DE COMPUTAÇÃO: {args.name}")
    print("=" * 70)
    
    # Criar servidor
    server = ComputationServer(args.name)
    
    # Configurar Pyro daemon
    daemon = Pyro5.api.Daemon(host=args.host, port=args.port)
    uri = daemon.register(server, objectId=args.name)
    
    print(f"\n✓ Servidor iniciado com sucesso!")
    print(f"  Nome: {args.name}")
    print(f"  URI: {uri}")
    print(f"  Host: {args.host}")
    print(f"  Porta: {args.port}")
    print(f"\nServidor aguardando conexões...")
    print("Pressione Ctrl+C para parar o servidor")
    print("=" * 70 + "\n")
    
    try:
        daemon.requestLoop()
    except KeyboardInterrupt:
        print("\n\n" + "=" * 70)
        print("Servidor encerrado pelo usuário")
        print("=" * 70)


if __name__ == "__main__":
    main()
