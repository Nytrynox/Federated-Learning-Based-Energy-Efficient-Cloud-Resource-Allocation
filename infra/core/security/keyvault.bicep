@description('The name of the Key Vault')
param name string

@description('The location for the Key Vault')
param location string = resourceGroup().location

@description('The principal ID to assign access policies to')
param principalId string

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: name
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: tenant().tenantId
    enabledForDeployment: false
    enabledForDiskEncryption: false
    enabledForTemplateDeployment: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    enableRbacAuthorization: true
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Assign Key Vault Administrator role to the principal
resource keyVaultRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: keyVault
  name: guid(keyVault.id, principalId, '00482a5a-887f-4fb3-b363-3b7fe8e74483')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '00482a5a-887f-4fb3-b363-3b7fe8e74483') // Key Vault Administrator
    principalId: principalId
    principalType: 'User'
  }
}

@description('The name of the Key Vault')
output name string = keyVault.name

@description('The URI of the Key Vault')
output endpoint string = keyVault.properties.vaultUri

@description('The resource ID of the Key Vault')
output resourceId string = keyVault.id
