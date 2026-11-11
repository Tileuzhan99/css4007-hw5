# Asks for a model API key (hidden input) and writes it into .env.
# The key is never printed and never leaves this machine.
#
#   .\set_key.ps1              # OpenAI (matches MODEL_PROVIDER=openai)
#   .\set_key.ps1 anthropic    # Anthropic
param([ValidateSet('openai', 'anthropic')][string]$Provider = 'openai')

$variable = if ($Provider -eq 'anthropic') { 'ANTHROPIC_API_KEY' } else { 'OPENAI_API_KEY' }
$model = if ($Provider -eq 'anthropic') { 'claude-opus-5-5' } else { 'gpt-5.6-luna' }

$envPath = Join-Path $PSScriptRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $PSScriptRoot ".env.example") $envPath
    Write-Host "Created .env from .env.example"
}

$secure = Read-Host -AsSecureString "Paste your $Provider API key and press Enter"
$key = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)).Trim()
if ([string]::IsNullOrWhiteSpace($key)) { Write-Error "No key entered; .env not changed."; exit 1 }

$kept = Get-Content $envPath | Where-Object {
    $_ -notmatch "^$variable=" -and $_ -notmatch '^MODEL_PROVIDER=' -and $_ -notmatch '^MODEL_NAME='
}
$lines = @("MODEL_PROVIDER=$Provider", "MODEL_NAME=$model", "$variable=$key") + $kept
[IO.File]::WriteAllLines($envPath, $lines, (New-Object Text.UTF8Encoding($false)))

Write-Host "Saved to .env (git-ignored). Checking that the key and model work..."
python -c @"
from dotenv import load_dotenv; load_dotenv()
from helpdesk import model
from langchain_core.messages import HumanMessage
from helpdesk.tools import LANGCHAIN_TOOLS
print('provider/model:', model.describe())
chat = model.build_chat_model().bind_tools(LANGCHAIN_TOOLS)
reply = chat.invoke([HumanMessage('Reply with the single word: ready')])
print('KEY OK, model answered:', (reply.content or '')[:60])
"@
