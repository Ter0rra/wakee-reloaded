# test_r2.py
import boto3
from botocore.exceptions import ClientError
import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path("..") / ".env"
load_dotenv(dotenv_path=env_path)

# Récupère les credentials
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_WR_IMG_BUCKET_NAME")

print("🔍 Test Cloudflare R2 Credentials")
print("=" * 50)

# Vérifie que les variables existent
print(f"\nR2_ACCOUNT_ID: {'✅' if R2_ACCOUNT_ID else '❌ Manquant'}")
print(f"R2_ACCESS_KEY_ID: {'✅' if R2_ACCESS_KEY_ID else '❌ Manquant'}")
print(f"R2_SECRET_ACCESS_KEY: {'✅' if R2_SECRET_ACCESS_KEY else '❌ Manquant'}")
print(f"R2_BUCKET_NAME: {'✅' if R2_BUCKET_NAME else '❌ Manquant'}")

if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
    print("\n❌ Credentials manquants !")
    exit(1)

# Test connexion
try:
    print(f"\n📡 Connexion à Cloudflare R2...")
    print(f"   Endpoint: https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com")
    
    s3_client = boto3.client(
        's3',
        endpoint_url=f'https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com',
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name='auto'
    )
    
    # Test 1 : Liste des buckets
    print(f"\n🪣 Test 1 : Liste des buckets...")
    response = s3_client.list_buckets()
    buckets = [bucket['Name'] for bucket in response['Buckets']]
    print(f"   Buckets disponibles : {buckets}")
    
    # Test 2 : Accès au bucket spécifique
    print(f"\n🔍 Test 2 : Accès au bucket '{R2_BUCKET_NAME}'...")
    s3_client.head_bucket(Bucket=R2_BUCKET_NAME)
    print(f"   ✅ Bucket '{R2_BUCKET_NAME}' accessible !")
    
    # Test 3 : Upload un fichier test
    print(f"\n📤 Test 3 : Upload fichier test...")
    test_content = b"Wakee test file"
    s3_client.put_object(
        Bucket=R2_BUCKET_NAME,
        Key='test/test.txt',
        Body=test_content
    )
    print(f"   ✅ Upload réussi : test/test.txt")
    
    # Test 4 : Liste des fichiers
    print(f"\n📋 Test 4 : Liste des fichiers dans le bucket...")
    response = s3_client.list_objects_v2(Bucket=R2_BUCKET_NAME, MaxKeys=5)
    if 'Contents' in response:
        print(f"   Fichiers trouvés : {len(response['Contents'])}")
        for obj in response['Contents'][:5]:
            print(f"     - {obj['Key']} ({obj['Size']} bytes)")
    else:
        print(f"   Bucket vide")
    
    # Test 5 : Supprimer le fichier test
    print(f"\n🗑️  Test 5 : Nettoyage...")
    s3_client.delete_object(Bucket=R2_BUCKET_NAME, Key='test/test.txt')
    print(f"   ✅ Fichier test supprimé")
    
    print("\n" + "=" * 50)
    print("🎉 TOUS LES TESTS RÉUSSIS !")
    print("=" * 50)
    print("\n✅ Tes credentials Cloudflare R2 sont corrects !")

except ClientError as e:
    error_code = e.response['Error']['Code']
    error_message = e.response['Error']['Message']
    
    print(f"\n❌ ERREUR : {error_code}")
    print(f"   Message : {error_message}")
    
    if error_code == 'NoSuchBucket':
        print(f"\n💡 Solution : Crée le bucket '{R2_BUCKET_NAME}' dans le dashboard R2")
    elif error_code == 'InvalidAccessKeyId':
        print(f"\n💡 Solution : Vérifie R2_ACCESS_KEY_ID")
    elif error_code == 'SignatureDoesNotMatch':
        print(f"\n💡 Solution : Vérifie R2_SECRET_ACCESS_KEY")
    else:
        print(f"\n💡 Solution : Vérifie tous les credentials")

except Exception as e:
    print(f"\n❌ ERREUR INATTENDUE : {e}")