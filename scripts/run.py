def main(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs):
    scaler = torch.cuda.amp.GradScaler()
  
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        train_loss = train(model, train_loader, optimizer, criterion, device, scaler)
        
   
        val_loss = validate(model, val_loader, criterion, device)
        

        scheduler.step(val_loss)
        
        print(f'Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}')
        print(f'Current learning rate: {optimizer.param_groups[0]["lr"]}')
