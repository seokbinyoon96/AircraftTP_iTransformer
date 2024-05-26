def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for data, target in progress_bar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            output, _ = model(data)  
            loss = criterion(output, target)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'Train Loss': loss.item()})
        
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(val_loader, desc='Validating', leave=False)
    with torch.no_grad():
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            
            with torch.cuda.amp.autocast():
                output, _ = model(data)  
                loss = criterion(output, target)
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Val Loss': loss.item()})
            
    return total_loss / len(val_loader)
