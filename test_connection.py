#!/usr/bin/env python3
"""
Script para testar conexão com servidores Pyro
"""
import Pyro5.api
import argparse
import sys


def test_connection(uri):
    """Testa conexão com um servidor Pyro."""
    print(f"\nTestando conexão com: {uri}")
    print("-" * 60)
    
    try:
        print("1. Criando proxy...")
        proxy = Pyro5.api.Proxy(uri)
        
        print("2. Tentando conectar...")
        proxy._pyroBind()
        
        print("3. Obtendo informações do servidor...")
        info = proxy.get_info()
        
        print("\n✓ CONEXÃO BEM-SUCEDIDA!")
        print(f"  Nome: {info['name']}")
        print(f"  Núcleos: {info['num_cores']}")
        
        proxy._pyroRelease()
        return True
        
    except Exception as e:
        print(f"\n✗ ERRO NA CONEXÃO:")
        print(f"  Tipo: {type(e).__name__}")
        print(f"  Mensagem: {e}")
        
        print("\nPossíveis soluções:")
        print("  1. Verifique se o servidor está rodando")
        print("  2. Confirme que a porta está correta")
        print("  3. Tente usar 127.0.0.1 em vez de localhost")
        print("  4. Verifique firewall/iptables")
        
        return False


def main():
    parser = argparse.ArgumentParser(description='Testar conexão com servidor Pyro')
    parser.add_argument('--uri', type=str, help='URI do servidor (ex: PYRO:server1@localhost:9090)')
    parser.add_argument('--host', type=str, default='localhost', help='Host do servidor')
    parser.add_argument('--port', type=int, default=9090, help='Porta do servidor')
    parser.add_argument('--name', type=str, default='server1', help='Nome do servidor')
    
    args = parser.parse_args()
    
    if args.uri:
        uri = args.uri
    else:
        uri = f"PYRO:{args.name}@{args.host}:{args.port}"
    
    print("=" * 60)
    print("TESTE DE CONEXÃO PYRO5")
    print("=" * 60)
    
    success = test_connection(uri)
    
    # Tentar alternativas se falhar
    if not success and 'localhost' in uri:
        print("\n" + "=" * 60)
        print("Tentando alternativa com 127.0.0.1...")
        print("=" * 60)
        
        uri_alt = uri.replace('localhost', '127.0.0.1')
        success = test_connection(uri_alt)
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Servidor acessível e funcionando!")
        sys.exit(0)
    else:
        print("✗ Não foi possível conectar ao servidor")
        sys.exit(1)


if __name__ == "__main__":
    main()