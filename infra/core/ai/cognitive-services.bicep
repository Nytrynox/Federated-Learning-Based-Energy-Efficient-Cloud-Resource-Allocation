@description('The name of the Cognitive Services account')
param name string

@description('The location for the Cognitive Services account')
param location string = resourceGroup().location

@description('The principal ID to assign roles to')
param principalId string

@description('The SKU for the Cognitive Services account')
param sku string = 'S0'

resource cognitiveServices 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: name
  location: location
  sku: {
    name: sku
  }
  kind: 'CognitiveServices'
  properties: {
    apiProperties: {}
    customSubDomainName: name
    networkAcls: {
      defaultAction: 'Allow'
    }
    publicNetworkAccess: 'Enabled'
  }
}

// Assign Cognitive Services User role to the principal
resource cognitiveServicesRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: cognitiveServices
  name: guid(cognitiveServices.id, principalId, 'a97b65f3-24c7-4388-baec-2e87135dc908')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908') // Cognitive Services User
    principalId: principalId
    principalType: 'User'
  }
}

@description('The name of the Cognitive Services account')
output name string = cognitiveServices.name

@description('The endpoint URL for the Cognitive Services account')
output endpoint string = cognitiveServices.properties.endpoint

@description('The resource ID of the Cognitive Services account')
output resourceId string = cognitiveServices.id
